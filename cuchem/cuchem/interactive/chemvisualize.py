# Copyright 2020 NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0

# TODO: separate loading of compounds from clustering of compounds; currently, loading is triggered by a call to clustering.
# TODO: separate fingerprinting from clustering; currently fingerprinting is triggered by a call to clustering.

import base64
import json
import logging
from io import StringIO
from pydoc import locate

import cupy
import dash
import cuml
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
from cuchemcommon.data.helper.chembldata import ChEmblData, IMP_PROPS
from cuchemcommon.utils.singleton import Singleton
from dash.dependencies import Input, Output, State, ALL
from flask import Response
from cuchem.decorator import LipinskiRuleOfFiveDecorator
from cuchem.decorator import MolecularStructureDecorator
from cuchem.utils import generate_colors, report_ui_error
from rdkit import Chem
from rdkit.Chem import Draw, PandasTools
from numba.cuda.libdevice import popcll

# Check if all of these are needed:
from cuchemcommon.fingerprint import MorganFingerprint, INTEGER_NBITS
import sys
import numpy as np
import pandas as pd
import cudf
import dask_cudf
from dask.distributed import wait
from rdkit import DataStructs, Chem
from rdkit.Chem.rdmolfiles import MolFromSmiles
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmilesFromSmiles
import time

logger = logging.getLogger(__name__)

main_fig_height = 700
CHEMBL_DB = '/data/db/chembl_27.db'
PAGE_SIZE = 10
DOT_SIZE = 5

LEVEL_TO_STYLE = {
    'info': {'color': 'black'},
    'warning': {'color': 'orange'},
    'error': {'color': 'red'}
}

PROP_DISP_NAME = {
    'chembl_id': 'ChEMBL Id',
    'mw_freebase': 'Molecular Weight (Free Base)',
    'alogp': 'AlogP',
    'hba': 'H-Bond Acceptors',
    'hbd': 'H-Bond Donors',
    'psa': 'Polar Surface Area',
    'rtb': 'Rotatable Area',
    'ro3_pass': 'Rule of 3 Passes',
    'num_ro5_violations': 'Lipinski Ro5 Violation',
    'cx_most_apka': 'Acidic pKa (ChemAxon)',
    'cx_most_bpka': 'Basic pKa (ChemAxon)',
    'cx_logp': 'logP (ChemAxon)',
    'cx_logd': 'LogD pKa (ChemAxon)',
    'molecular_species': 'Molecular Species',
    'full_mwt': 'MW (Full)',
    'aromatic_rings': 'Aromatic Rings',
    'heavy_atoms': 'Heavy Atoms',
    'qed_weighted': 'QED (Weighted)',
    'mw_monoisotopic': 'MW (Mono)',
    'full_molformula': 'Full Formula',
    'hba_lipinski': 'H-Bond Acceptors (Lipinski)',
    'hbd_lipinski': 'H-Bond Donors (Lipinski)',
    'num_lipinski_ro5_violations': 'Lipinski Ro5 Violations',
    'standard_inchi': 'Standard InChi',
    'standard_inchi_key': 'Standard InChi Key'
}

app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css', dbc.themes.BOOTSTRAP])


@app.server.route('/cheminfo/downloadSDF')
def download_sdf():
    logger.info('Exporting generated data...')

    vis = ChemVisualization()
    output = StringIO()

    valid_idx = []
    col_list = ['SMILES', 'Molecular Weight', 'LogP', 'H-Bond Donors', 'H-Bond Acceptors', 'Rotatable Bonds']
    for row, data in vis.generated_df.iterrows():
        mol = Chem.MolFromSmiles(data['SMILES'])
        if (mol is not None):
            valid_idx.append(row)

    valid_df = vis.generated_df.iloc[valid_idx]
    valid_df = valid_df[col_list]

    PandasTools.AddMoleculeColumnToFrame(valid_df, 'SMILES')
    PandasTools.WriteSDF(valid_df, output, properties=list(valid_df.columns))

    output.seek(0)

    return Response(
        output.getvalue(),
        mimetype="text/application",
        headers={"Content-disposition":
                     "attachment; filename=download.sdf"})

def popcll_wrapper(ip_col, op_col):
    for i, n in enumerate(ip_col):
        op_col[i] = popcll(n)

def popcll_wrapper_dask(df, ip_col, op_col):
    df = df.apply_rows(popcll_wrapper, incols = {ip_col: 'ip_col'}, outcols = {op_col: int}, kwargs = {})
    return df[op_col]

def intersection_wrapper(fp_int_col, op_col, query_fp_int):
    for i, fp_int in enumerate(fp_int_col):
        op_col[i] = popcll(fp_int & query_fp_int)

class ChemVisualization(metaclass=Singleton):

    def __init__(self, cluster_wf, fingerprint_radius=2, fingerprint_nBits=512):
        self.app = app
        self.cluster_wf = cluster_wf
        self.n_clusters = cluster_wf.n_clusters
        self.chem_data = ChEmblData()
        self.generated_df = None
        self.cluster_wf_cls = 'cuchem.wf.cluster.gpukmeansumap.GpuKmeansUmapHybrid'
        self.generative_wf_cls = 'cuchem.wf.generative.MegatronMolBART'

        self.fp_df = None # all fingerprints of all ChemBl compounds and their IDs as a pandas dataframe for use in compound similarity search on the CPU
        self.fingerprint_radius = fingerprint_radius
        self.fingerprint_nBits = fingerprint_nBits

        # Store colors to avoid plots changes colors on events such as
        # molecule selection, etc.
        self.cluster_colors = generate_colors(self.n_clusters)

        # Construct the UI
        self.app.layout = self.constuct_layout()

        # Register callbacks for selection inside main figure
        self.app.callback(
            [Output('selected_clusters', 'value'),
             Output('selected_point_cnt', 'children')],
            [Input('main-figure', 'clickData'),
             Input('main-figure', 'selectedData'),
             Input('bt_recluster_clusters', 'n_clicks'),
             Input('bt_recluster_points', 'n_clicks'),
             Input('northstar_cluster', 'children')],
            [State("selected_clusters", "value")])(self.handle_data_selection)

        # Register callbacks for buttons for reclustering selected data
        self.app.callback(
            [Output('main-figure', 'figure'),
             Output('northstar_cluster', 'children'),
             Output('refresh_moi_prop_table', 'children'),
             Output('recluster_error', 'children')],
            [Input('bt_recluster_clusters', 'n_clicks'),
             Input('bt_recluster_points', 'n_clicks'),
             Input('bt_north_star', 'n_clicks'),
             Input('sl_prop_gradient', 'value'),
             Input('sl_nclusters', 'value'),
             Input('refresh_main_fig', 'children'),
             Input('fingerprint_radius', 'value'),
             Input('fingerprint_nBits', 'value')],
            [State("selected_clusters", "value"),
             State("main-figure", "selectedData"),
             State('north_star', 'value'),
             State('refresh_moi_prop_table', 'children')])(self.handle_re_cluster)

        # Register callbacks for selection inside main figure to update module details
        self.app.callback(
            [Output('tb_selected_molecules', 'children'),
             Output('sl_mol_props', 'options'),
             Output('current_page', 'children'),
             Output('total_page', 'children'),
             Output('show_selected_mol', 'children'),
             Output('mol_selection_error', 'children')],
            [Input('main-figure', 'selectedData'),
             Input('sl_mol_props', 'value'),
             Input('bt_page_prev', 'n_clicks'),
             Input('bt_page_next', 'n_clicks'),
             Input('refresh_moi_prop_table', 'children')],
            [State('north_star', 'value'),
             State('current_page', 'children'),
             State('show_selected_mol', 'children'),
             State('sl_prop_gradient', 'value')])(self.handle_molecule_selection)

        self.app.callback(
            Output("refresh_main_fig", "children"),
            [Input("bt_reset", "n_clicks"),
             Input("bt_apply_wf", "n_clicks")],
            [State("refresh_main_fig", "children"),
             State("sl_wf", "value")])(self.handle_reset)

        self.app.callback(
            Output('north_star', 'value'),
            Input({'role': 'bt_star_candidate', 'chemblId': ALL, 'molregno': ALL}, 'n_clicks'),
            State('north_star', 'value'))(self.handle_mark_north_star)

        self.app.callback(
            [Output('error_msg', 'children'),
             Output('md_error', 'is_open')],
            [Input('recluster_error', 'children'),
             Input('interpolation_error', 'children'),
             Input('bt_close_err', 'n_clicks')])(self.handle_error)

        self.app.callback(
            Output('generation_candidates', 'children'),
            [Input({'role': 'bt_add_candidate', 'chemblId': ALL, 'molregno': ALL}, 'n_clicks'),
             Input('bt_reset_candidates', 'n_clicks'), ],
            State('generation_candidates', 'children'))(self.handle_add_candidate)

        self.app.callback(
            Output('analoguing_candidates', 'children'),
            [Input({'role': 'bt_analoguing_candidate', 'chemblId': ALL, 'molregno': ALL}, 'n_clicks')],
            State('analoguing_candidates', 'children'))(self.handle_analoguing_candidate)

        self.app.callback(
            Output('ckl_candidate_mol_id', 'options'),
            Input('generation_candidates', 'children'))(self.handle_construct_candidates)

        self.app.callback(
            Output('ckl_analoguing_mol_id', 'options'),
            Input('analoguing_candidates', 'children'))(self.handle_construct_candidates2)

        self.app.callback(
            [Output('ckl_candidate_mol_id', 'value'),
             Output('mk_selection_msg', 'children')],
            [Input('ckl_candidate_mol_id', 'value'),
             Input('rd_generation_type', 'value')])(self.handle_ckl_selection)

        self.app.callback(
            [Output('ckl_analoguing_mol_id', 'value')],
            [Input('ckl_analoguing_mol_id', 'value')])(self.handle_analoguing_ckl_selection)

        self.app.callback(
            [Output('section_generated_molecules_clustered', 'style'),
             Output('gen_figure', 'figure'),
             Output('table_generated_molecules', 'children'),
             Output('show_generated_mol', 'children'),
             Output('msg_generated_molecules', 'children'),
             Output('interpolation_error', 'children')],
            [Input("bt_generate", "n_clicks"), ],
            [State('sl_generative_wf', 'value'),
             State('ckl_candidate_mol_id', 'value'),
             State('n2generate', 'value'),
             State('extrap_compound_property', 'value'),
             State('extrap_cluster_number', 'value'),
             State('extrap_n_compounds', 'value'),
             State('extrap_step_size', 'value'),
             State('scaled_radius', 'value'),
             State('rd_generation_type', 'value'),
             State('show_generated_mol', 'children')])(self.handle_generation)

        self.app.callback(
            [Output('section_fitting', 'style'),
             Output('fitting_figure', 'figure')],
            [Input("bt_fit", "n_clicks"),],
            [State('sl_featurizing_wf', 'value'),
             State('fit_nn_compound_property', 'value'),
             State('fit_nn_train_cluster_number', 'value'),
             State('fit_nn_test_cluster_number', 'value'),
             State('fit_nn_hidden_layer_sizes', 'value'),
             State('fit_nn_activation_fn', 'value'),
             State('fit_nn_final_activation_fn', 'value'),
             State('fit_nn_max_epochs', 'value'),
             State('fit_nn_learning_rate', 'value'),
             State('fit_nn_weight_decay', 'value'),
             State('fit_nn_batch_size', 'value')])(self.handle_fitting)

        self.app.callback(
            [Output('section_analoguing', 'style'),
             Output('tb_analoguing', 'children')],
            [Input("bt_analoguing", "n_clicks"),],
            [State('ckl_analoguing_mol_id', 'value'),
             State('analoguing_n_analogues', 'value'),
             State('analoguing_threshold', 'value'),
             State('analoguing_type', 'value')])(self.handle_analoguing)

        self.app.callback(
            [Output('section_generated_molecules', 'style'),
             Output('section_selected_molecules', 'style'), ],
            [Input('show_generated_mol', 'children'),
             Input('show_selected_mol', 'children')])(self.handle_property_tables)

    def handle_add_candidate(self, bt_add_candidate,
                             bt_reset_candidates,
                             generation_candidates):
        comp_id, event_type = self._fetch_event_data()

        if comp_id == 'bt_reset_candidates' and event_type == 'n_clicks':
            return ''

        if event_type != 'n_clicks' or dash.callback_context.triggered[0]['value'] == 0:
            raise dash.exceptions.PreventUpdate

        selected_candidates = []

        if generation_candidates:
            selected_candidates = generation_candidates.split(",")

        comp_detail = json.loads(comp_id)
        selected_chembl_id = comp_detail['chemblId']

        if selected_chembl_id not in selected_candidates:
            selected_candidates.append(selected_chembl_id)

        return ','.join(selected_candidates)


    def handle_analoguing_candidate(self, bt_analoguing_candidate, analoguing_candidates):
        comp_id, event_type = self._fetch_event_data()
        if event_type != 'n_clicks' or dash.callback_context.triggered[0]['value'] == 0:
            raise dash.exceptions.PreventUpdate

        selected_candidates = []

        if analoguing_candidates:
            selected_candidates = analoguing_candidates.split(",")

        comp_detail = json.loads(comp_id)
        selected_chembl_id = comp_detail['chemblId']

        if selected_chembl_id not in selected_candidates:
            selected_candidates.append(selected_chembl_id)
        return ','.join(selected_candidates)

    def _fetch_event_data(self):
        if not dash.callback_context.triggered:
            raise dash.exceptions.PreventUpdate
        prop_id = dash.callback_context.triggered[0]['prop_id']
        split_at = prop_id.rindex('.')
        return [prop_id[:split_at], prop_id[split_at + 1:]]

    def handle_property_tables(self, show_generated_mol, show_selected_mol):
        comp_id, event_type = self._fetch_event_data()

        if comp_id == 'show_selected_mol' and event_type == 'children':
            return {'display': 'none'}, {'display': 'block', 'width': '100%'}
        elif comp_id == 'show_generated_mol' and event_type == 'children':
            return {'display': 'block', 'width': '100%'}, {'display': 'none'}
        return dash.no_update, dash.no_update

    @report_ui_error(4)
    def handle_generation(
        self, bt_generate, sl_generative_wf, ckl_candidate_mol_id, n2generate, 
        extrap_compound_property, extrap_cluster_number, extrap_n_compounds, extrap_step_size, 
        scaled_radius, rd_generation_type, show_generated_mol
    ):    
        comp_id, event_type = self._fetch_event_data()

        chembl_ids = []
        if comp_id == 'bt_generate' and event_type == 'n_clicks':
            chembl_ids = ckl_candidate_mol_id
        else:
            return dash.no_update, dash.no_update

        self.generative_wf_cls = sl_generative_wf
        wf_class = locate(self.generative_wf_cls)
        generative_wf = wf_class()
        n2generate = int(n2generate)
        scaled_radius = float(scaled_radius)

        if rd_generation_type == 'SAMPLE':
            if chembl_ids == None or len(chembl_ids) == 0:
                raise ValueError('Please select at-least one molecule for Sampling.')
            self.generated_df = generative_wf.find_similars_smiles_by_id(chembl_ids,
                                                                         num_requested=n2generate,
                                                                         scaled_radius=scaled_radius,
                                                                         force_unique=True,
                                                                         sanitize=True)
        elif rd_generation_type == 'EXTRAPOLATE':
            self.generated_df = generative_wf.extrapolate_from_cluster(self.cluster_wf.df_embedding,
                                                                       compound_property=extrap_compound_property,
                                                                       cluster_id=extrap_cluster_number,
                                                                       n_compounds_to_transform=extrap_n_compounds,									
                                                                       num_points=n2generate,
                                                                       step_size=extrap_step_size,
                                                                       scaled_radius=scaled_radius,
                                                                       force_unique=False)#True)                                                                         
        else:
            if chembl_ids == None or len(chembl_ids) < 2:
                raise ValueError('Please select at-least two molecules for Interpolation.')
            self.generated_df = generative_wf.interpolate_by_id(chembl_ids,
                                                                num_points=n2generate,
                                                                scaled_radius=scaled_radius,
                                                                force_unique=True,
                                                                sanitize=True)

        if show_generated_mol is None:
            show_generated_mol = 0
        show_generated_mol += 1
        # Add other useful attributes to be added for rendering
        self.generated_df = MolecularStructureDecorator().decorate(self.generated_df)
        self.generated_df = LipinskiRuleOfFiveDecorator().decorate(self.generated_df)
        self.generated_df = self.generated_df[ ~self.generated_df['invalid'] ].reset_index(drop=True).drop(columns=['invalid'])
        if len(self.generated_df) == 0:
            logger.info("None of the generated smiles yielded valid molecules!!!")
            return dash.no_update, dash.no_update

        # Note: we are not allowing fingerprint specification to change here because we want to see the results on the same PCA / UMAP as the original figure
        # TODO: make this clear in the UI
        fps = MorganFingerprint(
            radius=self.fingerprint_radius, nBits=self.fingerprint_nBits
        ).transform(self.generated_df, smiles_column='SMILES')
        df_fp = pd.DataFrame(fps, dtype='float32')
        self.generated_df = pd.concat([self.generated_df, df_fp], axis=1)  
        df_fp=cudf.from_pandas(df_fp)
        df_fp['id'] = list(map(str, self.generated_df['id']))
        df_fp['cluster'] = list(map(int, self.generated_df['Generated'])) # This controls the color
        n_generated =  self.generated_df['Generated'].sum()
        if n_generated < len(self.generated_df) / 2:
            # Highlight the generated compounds
            north_stars = ','.join(list(df_fp[ self.generated_df['Generated'] ]['id'].values_host))
        else:
            # Highlight the source compound(s)
            north_stars = ','.join(list(df_fp[ ~self.generated_df['Generated'] ]['id'].values_host))     

        # TODO: check if all these lines are necessary!
        chunksize=max(10, int(df_fp.shape[0] * 0.1))
        df_embedding = dask_cudf.from_cudf(df_fp, chunksize=chunksize)
        df_embedding = df_embedding.reset_index()
        cluster_col = df_embedding['cluster']
        df_embedding, prop_series = self.cluster_wf._remove_non_numerics(df_embedding)
        prop_series['cluster'] = cluster_col
        n_molecules, n_obs = df_embedding.compute().shape # needed?
        #if hasattr(df_embedding, 'compute'):
        #    df_embedding = df_embedding.compute()

        if isinstance(self.cluster_wf.pca, cuml.PCA) and isinstance(df_embedding, dask_cudf.DataFrame):
            # Trying to accommodate the GpuKmeansUmapHybrid workflow
            df_embedding = df_embedding.compute()
        df_embedding = self.cluster_wf.pca.transform(df_embedding)
        if hasattr(df_embedding, 'persist'):
            df_embedding = df_embedding.persist()
            wait(df_embedding)
        Xt = self.cluster_wf.umap.transform(df_embedding)
        df_embedding['x'] = Xt[0]
        df_embedding['y'] = Xt[1]

        for col in prop_series.keys():
            sys.stdout.flush()
            df_embedding[col] = prop_series[col]#.compute()

        fig, northstar_cluster = self.create_graph(df_embedding, north_stars=north_stars)

        # Create Table header
        table_headers = []
        all_columns = self.generated_df.columns.to_list()
        columns_in_table = [
            col_name 
            for col_name in self.generated_df.columns.to_list()
            if (not isinstance(col_name, int)) and (not col_name.startswith('fp')) and not ('embeddings' in col_name)
        ]
        # TODO: factor this into a separate function: build table from dataframe
        for column in columns_in_table:
            table_headers.append(html.Th(column, style={'fontSize': '150%', 'text-align': 'center'}))
        prop_recs = [html.Tr(table_headers, style={'background': 'lightgray'})]
        for row_idx in range(self.generated_df.shape[0]):
            td = []
            try:
                col_pos = all_columns.index('Chemical Structure')
                col_data = self.generated_df.iat[row_idx, col_pos]
                if 'value' in col_data and col_data['value'] == 'Error interpreting SMILES using RDKit':
                    continue
            except ValueError:
                pass
            for col_name in columns_in_table:
                col_id = all_columns.index(col_name)
                col_data = self.generated_df.iat[row_idx, col_id]
                col_level = 'info'
                if isinstance(col_data, dict):
                    col_value = col_data['value']
                    if 'level' in col_data:
                        col_level = col_data['level']
                else:
                    col_value = col_data
                if isinstance(col_value, str) and col_value.startswith('data:image/png;base64,'):
                    td.append(html.Td(html.Img(src=col_value)))
                else:
                    td.append(
                        html.Td(str(col_value), style=LEVEL_TO_STYLE[col_level].update({'maxWidth': '100px', 'wordWrap':'break-word'})))
            prop_recs.append(html.Tr(td))

        return {'display': 'inline'}, fig, html.Table(
            prop_recs, style={'width': '100%', 'margin': 12, 'border': '1px solid lightgray'}
            ), show_generated_mol, dash.no_update


    @report_ui_error(3)
    def handle_fitting(
        self, bt_fit, sl_featurizing_wf, 
        fit_nn_compound_property, fit_nn_train_cluster_number, fit_nn_test_cluster_number, fit_nn_hidden_layer_sizes, fit_nn_activation_fn, fit_nn_final_activation_fn, 
        fit_nn_max_epochs, fit_nn_learning_rate, fit_nn_weight_decay, fit_nn_batch_size
    ):
        comp_id, event_type = self._fetch_event_data()
        sys.stdout.flush()
        if (comp_id != 'bt_fit') or (event_type != 'n_clicks'):
            return dash.no_update, dash.no_update
        self.featurizing_wf_cls = sl_featurizing_wf
        wf_class = locate(self.featurizing_wf_cls)
        featurizing_wf = wf_class()

        df = featurizing_wf.fit_nn(
            self.cluster_wf.df_embedding,
            compound_property=fit_nn_compound_property,
            cluster_id_train=fit_nn_train_cluster_number,
            cluster_id_test=fit_nn_test_cluster_number,
            hidden_layer_sizes=list(map(int, fit_nn_hidden_layer_sizes.split(','))) if fit_nn_hidden_layer_sizes != '' else [],
            activation_fn=fit_nn_activation_fn,
            final_activation_fn=fit_nn_final_activation_fn,
            max_epochs=int(fit_nn_max_epochs),
            learning_rate=float(fit_nn_learning_rate),
            weight_decay=float(fit_nn_weight_decay),
            batch_size=int(fit_nn_batch_size)
        )
        sys.stdout.flush()
        fig = self.create_plot(df, fit_nn_compound_property)
        return {'display': 'inline'}, fig

    @report_ui_error(3)
    def handle_analoguing(
        self, bt_analoguing, analoguing_mol_id, analoguing_n_analogues, analoguing_threshold, analoguing_type,

    ):
        comp_id, event_type = self._fetch_event_data()
        sys.stdout.flush()
        if (comp_id != 'bt_analoguing') or (event_type != 'n_clicks'):
            return dash.no_update, dash.no_update

        # Compute fingerprints once for all input database compounds (already done when input data would have been clustered)
        if 'canonical_smiles' in self.cluster_wf.df_embedding:
            smiles_column = 'canonical_smiles'
        else:
            smiles_columns = 'SMILES'

        if self.fp_df is None: 
            # Note: CPU-based workflow is no longer needed, can be removed
            logger.info(f'CPU-based similarity search: self.fp_df not set')
            # First move the smiles to the CPU:
            if isinstance(self.cluster_wf.df_embedding, dask_cudf.DataFrame):
                smiles_df = self.cluster_wf.df_embedding[[smiles_column, 'id']].map_partitions(cudf.DataFrame.to_pandas)
            elif isinstance(self.cluster_wf.df_embedding, cudf.DataFrame):
                smiles_df = self.cluster_wf.df_embedding[[smiles_column, 'id']].to_pandas()
            else:
                smiles_df = self.cluster_wf.df_embedding[[smiles_column, 'id']]
            # Then compute fingerprints on the CPU using RDKit:
            if 'fp' not in self.cluster_wf.df_embedding.columns:           
                logger.info(f'Computing fingerprints with radius={self.fingerprint_radius}, nBits={self.fingerprint_nBits}...')
                _, v = MorganFingerprint(radius=self.fingerprint_radius, nBits=self.fingerprint_nBits).transform(
                    smiles_df, smiles_column=smiles_column, return_fp=True, raw=True)
            else:
                logger.info(f'Fingerprints already available')
                if hasattr(self.cluster_wf.df_embedding, 'compute'):
                    v = list(self.cluster_wf.df_embedding['fp'].compute().to_pandas())
                else:
                    v = list(self.cluster_wf.df_embedding['fp'])
            # This pandas dataframe has the fingerprints in the fp column:
            self.fp_df = pd.DataFrame({
                'fp': v, 
                smiles_column: smiles_df[smiles_column], #list(self.cluster_wf.df_embedding[smiles_column].compute().to_pandas()), #smiles_df[smiles_column], 
                'id': smiles_df['id'], #list(self.cluster_wf.df_embedding['id'].compute().to_pandas())
            })         

        if hasattr(self.cluster_wf.df_embedding, 'persist'):
            self.cluster_wf.df_embedding = self.cluster_wf.df_embedding.persist()
            wait(self.cluster_wf.df_embedding)

        if 'pc' not in self.cluster_wf.df_embedding.columns:
            # Pre-computing the popcounts for all compounds in the database for use in GPU-based similarity search:
            t0 = time.time()
            self.cluster_wf.df_embedding['op_col'] = 0
            self.cluster_wf.df_embedding['pc'] = 0
            n_fp_cols = 0
            for col in self.cluster_wf.df_embedding.columns:
                if (type(col) == str) and col.startswith('fp') and (len(col) > 2):
                    n_fp_cols += 1
                    self.cluster_wf.df_embedding = self.cluster_wf.df_embedding.apply_rows(
                        popcll_wrapper, incols = {col: 'ip_col'}, outcols = {'op_col': int}, kwargs = {})
                    # More complex syntax was not necessary:
                    #self.cluster_wf.df_embedding['op_col'] = self.cluster_wf.df_embedding.map_partitions(popcll_wrapper_dask, col, 'op_col') #lambda df: df = df.apply_rows(popcll_wrapper, incols = {col: 'ip_col'}, outcols = {'op_col': int}, kwargs = {}))
                    self.cluster_wf.df_embedding['pc'] += self.cluster_wf.df_embedding['op_col']
            if hasattr(self.cluster_wf.df_embedding, 'persist'):
                self.cluster_wf.df_embedding = self.cluster_wf.df_embedding.persist()
                wait(self.cluster_wf.df_embedding)
            t1 = time.time()
            logger.info(f'Time to compute partial popcounts ({n_fp_cols} fp columns): {t1 - t0}:\n{self.cluster_wf.df_embedding["pc"].head()}')

        # Prepare the query compound:
        logger.info(f'analoguing_mol_id={analoguing_mol_id}')
        molregno = self.chem_data.fetch_molregno_by_chemblId(
            [analoguing_mol_id])[0][0]
        props, selected_molecules = self.chem_data.fetch_props_by_molregno([molregno])
        query_smiles = selected_molecules[0][props.index('canonical_smiles')]
        query_fp =  MorganFingerprint(radius=self.fingerprint_radius, nBits=self.fingerprint_nBits).transform(
            pd.DataFrame({'smiles': [query_smiles]}), smiles_column='smiles', return_fp=True, raw=True)[1][0]
        query_fps = query_fp.ToBitString()
        query_fp_ints = [int(query_fps[i: i + INTEGER_NBITS], 2) for i in range(0, self.fingerprint_nBits, INTEGER_NBITS)]
        query_pc = sum(bin(x).count('1') for x in query_fp_ints)

        # GPU-based workflow for similarity computation
        # Tanimoto = popcount(intersection) / ( popcount(query) + popcount(compound) - popcount(intersection) )
        # Sine the fingerprint is stored as a list of int64s in separate columns
        if 'op_col' in self.cluster_wf.df_embedding:
            self.cluster_wf.df_embedding = self.cluster_wf.df_embedding.drop(columns=['op_col'])
        if 'n_intersection' in self.cluster_wf.df_embedding:
            self.cluster_wf.df_embedding = self.cluster_wf.df_embedding.drop(columns=['n_intersection'])
        #self.cluster_wf.df_embedding['op_col'] = 0
        self.cluster_wf.df_embedding['n_intersection'] = 0
        t4 = time.time()
        for i in range(0, self.fingerprint_nBits, INTEGER_NBITS):
            fp_num = i // INTEGER_NBITS
            self.cluster_wf.df_embedding = self.cluster_wf.df_embedding.apply_rows(
                intersection_wrapper, incols={f'fp{fp_num}': 'fp_int_col'}, 
                outcols={'op_col': int}, kwargs={'query_fp_int': query_fp_ints[fp_num]})
            #self.cluster_wf.df_embedding = self.cluster_wf.df_embedding.persist()
            #wait(self.cluster_wf.df_embedding)
            self.cluster_wf.df_embedding['n_intersection'] += self.cluster_wf.df_embedding['op_col']

        self.cluster_wf.df_embedding['n_union'] = self.cluster_wf.df_embedding['pc'] - self.cluster_wf.df_embedding['n_intersection'] + query_pc
        self.cluster_wf.df_embedding['similarity'] = self.cluster_wf.df_embedding['n_intersection'] / self.cluster_wf.df_embedding['n_union']
        self.cluster_wf.df_embedding = self.cluster_wf.df_embedding.persist()
        wait(self.cluster_wf.df_embedding)
        t5 = time.time()
        t0 = time.time()
        self.fp_df['similarity_cpu'] = self.fp_df['fp'].apply(lambda x: DataStructs.FingerprintSimilarity(query_fp, x))

        if 'similarity_cpu' in self.cluster_wf.df_embedding:
            self.cluster_wf.df_embedding = self.cluster_wf.df_embedding.drop(columns=['similarity_cpu'])
        self.cluster_wf.df_embedding = self.cluster_wf.df_embedding.merge(
            dask_cudf.from_cudf(
                cudf.from_pandas(self.fp_df[['id', 'similarity_cpu']]), 
                npartitions = self.cluster_wf.df_embedding.npartitions
            ),
            on='id',
            how='left'
        ).reset_index(drop=True)

        t1 = time.time()
        logger.info(f'Fingerprint length={self.fingerprint_nBits}: GPU-Method: {t5 - t4}, CPU-Method: {t1 - t0}')

        #self.analoguing_df = self.fp_df[ self.fp_df['similarity_cpu'] >= float(analoguing_threshold) ]
        self.analoguing_df = self.cluster_wf.df_embedding[ self.cluster_wf.df_embedding['similarity'] >= float(analoguing_threshold) ]
        drop_columns = [
            col
            for col in self.analoguing_df.columns
            if (type(col) == int) or col.startswith('fp') or (col in ['x', 'y', 'cluster', 'op_col', 'pc', 'n_intersection', 'n_union', 'transformed_smiles']) 
        ]
        self.analoguing_df = self.analoguing_df.drop(columns=drop_columns).compute().to_pandas() # dask_cudf --> cudf --> pandas (CPU!)
        if analoguing_type in ['scaffold', 'superstructure']:
            if analoguing_type == 'scaffold':
                # Only include compounds that have the same murcko scaffold as the query compound
                query_scaffold_mol = MolFromSmiles(MurckoScaffoldSmilesFromSmiles(query_smiles))
            else: #analoguing_type == 'superstructure':
                # Only include compounds that are superstructures of the query compound
                query_scaffold_mol = MolFromSmiles(query_smiles)
            self.analoguing_df['mol'] = self.analoguing_df[smiles_column].apply(MolFromSmiles)
            self.analoguing_df.dropna(subset=['mol'], inplace=True)
            self.analoguing_df = self.analoguing_df[ self.analoguing_df['mol'].apply(lambda x: x.HasSubstructMatch(query_scaffold_mol)) ]
            self.analoguing_df.drop(columns=['mol'], inplace=True)
        self.analoguing_df = self.analoguing_df.nlargest(int(analoguing_n_analogues), 'similarity') 
        self.analoguing_df.reset_index(drop=True, inplace=True)
        #self.analoguing_df = dask_cudf.from_cudf(self.analoguing_df, npartitions=self.cluster_wf.df_embedding.npartitions) # going back to dask for a reason?
        # TODO: we are presuming the IDs are the same but there is no guarantee since we added code to generate dummy IDs based on indices elsewhere.
 
        # Needed only for CPU-based workflow
        #self.analoguing_df = self.analoguing_df.merge(self.cluster_wf.df_embedding, on='id').compute().reset_index(drop=True).to_pandas()
        # Add other useful attributes to be added for rendering
        smiles_idx = self.analoguing_df.columns.to_list().index(smiles_column)
        self.analoguing_df = MolecularStructureDecorator().decorate(self.analoguing_df, smiles_col=smiles_idx)
        #self.analoguing_df = LipinskiRuleOfFiveDecorator().decorate(self.analoguing_df, smiles_col=smiles_idx)
        self.analoguing_df = self.analoguing_df.sort_values('similarity', ascending=False)
        # Create Table header
        table_headers = []
        all_columns = self.analoguing_df.columns.to_list()
        columns_in_table = [
            col_name 
            for col_name in self.analoguing_df.columns.to_list()
            if (not isinstance(col_name, int)) and (not col_name.startswith('fp'))
        ]
        # TODO: factor this into a separate function: build table from dataframe
        for column in columns_in_table:
            table_headers.append(html.Th(column, style={'fontSize': '150%', 'text-align': 'center'}))
        prop_recs = [html.Tr(table_headers, style={'background': 'lightgray'})]
        for row_idx in range(self.analoguing_df.shape[0]):
            td = []
            try:
                col_pos = all_columns.index('Chemical Structure')
                col_data = self.analoguing_df.iat[row_idx, col_pos]
                if 'value' in col_data and col_data['value'] == 'Error interpreting SMILES using RDKit':
                    continue
            except ValueError:
                pass
            for col_name in columns_in_table:
                col_id = all_columns.index(col_name)
                col_data = self.analoguing_df.iat[row_idx, col_id]
                col_level = 'info'
                if isinstance(col_data, dict):
                    col_value = col_data['value']
                    if 'level' in col_data:
                        col_level = col_data['level']
                else:
                    col_value = col_data
                if isinstance(col_value, str) and col_value.startswith('data:image/png;base64,'):
                    td.append(html.Td(html.Img(src=col_value)))
                else:
                    td.append(html.Td(str(col_value), style=LEVEL_TO_STYLE[col_level].update({'maxWidth': '100px', 'wordWrap':'break-word'})))

            prop_recs.append(html.Tr(td))


        # venkat: 
        return {'display': 'inline'}, html.Table(prop_recs, style={'width': '100%', 'margin': 12, 'border': '1px solid lightgray'})
        # dev:
        #return html.Table(prop_recs, style={'width': '100%',
        #                                    'border': '1px solid lightgray'}), \
        #       show_generated_mol, \
        #       msg_generated_molecules, \
        #       dash.no_update

    def handle_ckl_selection(self, ckl_candidate_mol_id, rd_generation_type):
        selection_msg = '**Please Select Two Molecules**'
        selection_cnt = 2

        if rd_generation_type == 'SAMPLE':
            selection_msg = '**Please Select One Molecule**'
            selection_cnt = 1
        elif rd_generation_type == 'EXTRAPOLATE':
            # TO DO: one cluster and one property have to be provided
            selection_msg = '**Please Select Zero Molecules (specify cluster above, instead)**'
            selection_cnt = 0
        if ckl_candidate_mol_id and len(ckl_candidate_mol_id) > selection_cnt:
            ckl_candidate_mol_id = ckl_candidate_mol_id[selection_cnt * -1:]

        return ckl_candidate_mol_id, selection_msg

    def handle_analoguing_ckl_selection(self, ckl_analoguing_mol_id):
        if ckl_analoguing_mol_id and len(ckl_analoguing_mol_id) > 1:
            # Allow only one compound to be chosen for analoguing
            ckl_analoguing_mol_id = ckl_analoguing_mol_id[-1:]

        return ckl_analoguing_mol_id

    def handle_construct_candidates(self, north_star):
        if not north_star:
            return []

        options = [{'label': i.strip(), 'value': i.strip()} for i in north_star.split(',')]
        return options

    def handle_construct_candidates2(self, north_star):
        if not north_star:
            return []

        options = [{'label': i.strip(), 'value': i.strip()} for i in north_star.split(',')]
        return options

    def handle_reset(self, bt_reset, bt_apply_wf, refresh_main_fig, sl_wf):
        comp_id, event_type = self._fetch_event_data()

        if comp_id == 'bt_apply_wf' and event_type == 'n_clicks':
            if self.cluster_wf_cls != sl_wf:
                self.cluster_wf_cls = sl_wf
                wf_class = locate(self.cluster_wf_cls)
                self.cluster_wf = wf_class()
            else:
                raise dash.exceptions.PreventUpdate

        if refresh_main_fig is None:
            refresh_main_fig = 1
        else:
            refresh_main_fig = int(refresh_main_fig)

        # Change the refresh variable to force main-figure refresh
        return refresh_main_fig + 1

    def recluster(self, filter_values=None, filter_column=None, reload_data=False):
        self.cluster_wf.n_clusters = self.n_clusters
        if reload_data:
            return self.cluster_wf.cluster(
                fingerprint_radius=self.fingerprint_radius, fingerprint_nBits=self.fingerprint_nBits
            )
        else:
            return self.cluster_wf.recluster(
                filter_column, 
                filter_values,
                n_clusters=self.n_clusters,
                fingerprint_radius=self.fingerprint_radius, 
                fingerprint_nBits=self.fingerprint_nBits
            )

    def recluster_selection(
        self,
        filter_value=None,
        filter_column=None,
        gradient_prop=None,
        north_stars=None,
        reload_data=False,
        recluster_data=True,
        color_col='cluster', 
        fingerprint_radius=2, 
        fingerprint_nBits=512
    ):

        if recluster_data or self.cluster_wf.df_embedding is None:
            self.fingerprint_nBits = fingerprint_nBits
            self.fingerprint_radius = fingerprint_radius
            df_embedding = self.recluster(
                filter_values=filter_value,
                filter_column=filter_column,
                reload_data=reload_data
            )
        else:
            # Can use previous embedding only if fingerprint has not changed
            df_embedding = self.cluster_wf.df_embedding

        return self.create_graph(df_embedding,
                                 color_col=color_col,
                                 gradient_prop=gradient_prop,
                                 north_stars=north_stars)

    def create_graph(self, ldf, color_col='cluster', north_stars=None, gradient_prop=None):
        sys.stdout.flush()
        fig = go.Figure(layout={'colorscale': {}})

        # Filter out relevant columns in this method.
        if hasattr(ldf, 'compute'):
            relevant_cols = ['id', 'x', 'y', 'cluster']
            if gradient_prop:
                relevant_cols.append(gradient_prop)
            if color_col == 'cluster':
                relevant_cols.append(color_col)

            ldf = ldf.iloc[:, ldf.columns.isin(relevant_cols)]
            ldf = ldf.compute()

        moi_molregno = []
        if north_stars:
            moi_molregno = north_stars.split(",") #list(map(int, north_stars.split(",")))

        moi_filter = ldf['id'].isin(moi_molregno)

        # Create a map with MoI and cluster to which they belong
        northstar_cluster = []
        if gradient_prop is not None:
            cmin = ldf[gradient_prop].min()
            cmax = ldf[gradient_prop].max()

            # Compute size of northstar and normal points
            df_shape = moi_filter.copy()
            df_size = (moi_filter * 18) + DOT_SIZE
            df_shape = df_shape * 2

            x_data = ldf['x']
            y_data = ldf['y']
            cluster = ldf['cluster']
            customdata = ldf['id']
            grad_prop = ldf[gradient_prop]
            textdata = cupy.asarray([
                f'C-{c}_ID-{cid}' for c, cid in zip(cdf['cluster'].to_array(), cdf['id'].to_array()) ])

            if self.cluster_wf.is_gpu_enabled():
                x_data = x_data.to_array()
                y_data = y_data.to_array()
                cluster = cluster.to_array()
                grad_prop = grad_prop.to_array()
                customdata = customdata.to_array()
                df_size = cupy.asnumpy(df_size)
                df_shape = cupy.asnumpy(df_shape)

            fig.add_trace(go.Scattergl({
                'x': x_data,
                'y': y_data,
                'text': cluster,
                'customdata': customdata,
                'mode': 'markers',
                'showlegend': False,
                'marker': {
                    'size': df_size,
                    'symbol': df_shape,
                    'color': grad_prop,
                    'colorscale': 'Viridis',
                    'showscale': True,
                    'cmin': cmin,
                    'cmax': cmax,
                    'name': customdata
                }
            }))
        else:
            clusters = ldf[color_col].unique()
            if self.cluster_wf.is_gpu_enabled():
                clusters = clusters.values_host

            northstar_df = ldf[moi_filter]
            scatter_traces = []
            for cluster_id in clusters:
                cdf = ldf.query('cluster == ' + str(cluster_id))

                df_size = cdf['id'].isin(northstar_df['id'])
                moi_present = False
                if df_size.unique().shape[0] > 1:
                    northstar_cluster.append(str(cluster_id))
                    moi_present = True

                # Compute size of northstar and normal points
                df_shape = df_size.copy()
                df_size = (df_size * 18) + DOT_SIZE
                df_shape = df_shape * 2
                x_data = cdf['x']
                y_data = cdf['y']
                cluster = cdf['cluster']
                customdata = cdf['id']
                textdata = [ f'C-{c}_ID-{cid}' for c, cid in zip(cdf['cluster'].to_array(), cdf['id'].to_array()) ]
                sys.stdout.flush()

                if self.cluster_wf.is_gpu_enabled():
                    x_data = x_data.to_array()
                    y_data = y_data.to_array()
                    cluster = cluster.to_array()
                    customdata = customdata.to_array()
                    df_size = cupy.asnumpy(df_size)
                    df_shape = cupy.asnumpy(df_shape)

                scatter_trace = go.Scattergl({
                    'x': x_data,
                    'y': y_data,
                    'text': textdata,
                    #'text': cluster,
                    'customdata': customdata,
                    'name': 'Cluster ' + str(cluster_id),
                    'mode': 'markers',
                    'marker': {
                        'size': df_size,
                        'symbol': df_shape,
                        'color': self.cluster_colors[
                            int(cluster_id) % len(self.cluster_colors)],
                    },
                })
                if moi_present:
                    # save to add later. This is to ensure the scatter is on top
                    scatter_traces.append(scatter_trace)
                else:
                    fig.add_trace(scatter_trace)

            for scatter_trace in scatter_traces:
                fig.add_trace(scatter_trace)

        # Change the title to indicate type of H/W in use
        f_color = 'green' if self.cluster_wf.is_gpu_enabled() else 'blue'

        fig.update_layout(
            showlegend=True, clickmode='event', height=main_fig_height,
            title='Clusters', dragmode='select',
            title_font_color=f_color,
            annotations=[
                dict(x=0.5, y=-0.07, showarrow=False, text='x',
                     xref="paper", yref="paper"),
                dict(x=-0.05, y=0.5, showarrow=False, text="y",
                     textangle=-90, xref="paper", yref="paper")])
        del ldf
        return fig, northstar_cluster

    def create_plot(self, df, compound_property):
        """
        Expects df to have x, y, cluster and train_set columns
        """
        fig = go.Figure(layout={'colorscale': {}})
        scatter_trace = go.Scattergl({
            'x': df['x'],
            'y': df['y'],
            'text': [ f'C-{c}_ID-{cid}' for c, cid in zip(df['cluster'], df['id']) ],
            'customdata': df['id'],
            'mode': 'markers',
            'marker': {
                'size': DOT_SIZE,
                'symbol': df['train_set'].apply(lambda x: 0 if x else 1),
                'color': df['cluster'].apply(lambda x: self.cluster_colors[x % len(self.cluster_colors)]),
            },
        })
        fig.add_trace(scatter_trace)
        # Change the title to indicate type of H/W in use
        f_color = 'green' if self.cluster_wf.is_gpu_enabled() else 'blue'
        fig.update_layout(
            showlegend=True, clickmode='event', height=main_fig_height,
            title=f'{PROP_DISP_NAME[compound_property]} Prediction', dragmode='select',
            title_font_color=f_color,
            annotations=[
                dict(x=0.5, y=-0.07, showarrow=False, text='Actual',
                     xref="paper", yref="paper"),
                dict(x=-0.05, y=0.5, showarrow=False, text="Predicted",
                     textangle=-90, xref="paper", yref="paper")])
        return fig

    def start(self, host=None, port=5000):
        return self.app.run_server(
            debug=False, use_reloader=False, host=host, port=port)

    def href_ify(self, molregno):
        return html.A(molregno,
                      href='https://www.ebi.ac.uk/chembl/compound_report_card/' + str(molregno),
                      target='_blank')

    def construct_molecule_detail(self, selected_points, display_properties,
                                  page, pageSize=10, chembl_ids=None):

        # Create Table header
        table_headers = [
            html.Th("Chemical Structure", style={'width': '30%', 'fontSize': '150%', 'text-align': 'center'}),
            html.Th("SMILES", style={'maxWidth': '100px', 'fontSize': '150%', 'text-align': 'center'})]
        for prop in display_properties:
            if prop in PROP_DISP_NAME:
                table_headers.append(html.Th(PROP_DISP_NAME[prop], style={'fontSize': '150%', 'text-align': 'center'}))

        if chembl_ids:
            table_headers.append(html.Th('ChEMBL', style={'fontSize': '150%', 'text-align': 'center'}))
        else:
            table_headers.append(html.Th("", style={'width': '10px'}))
        table_headers.append(html.Th("", style={'width': '10px'}))

        prop_recs = [html.Tr(table_headers, style={'background': 'lightgray'})]

        if chembl_ids:
            props, selected_molecules = self.chem_data.fetch_props_by_chembl(chembl_ids)
        elif selected_points:
            selected_molregno = []
            for point in selected_points['points'][((page - 1) * pageSize): page * pageSize]:
                if 'customdata' in point:
                    molregid = point['customdata']
                    selected_molregno.append(molregid)
            props, selected_molecules = self.chem_data.fetch_props_by_molregno(
                selected_molregno)
        else:
            return None, None

        all_props = []
        for k in props:
            if k in PROP_DISP_NAME:
                all_props.append({"label": PROP_DISP_NAME[k], "value": k})

        for selected_molecule in selected_molecules:
            td = []
            selected_chembl_id = selected_molecule[1]
            smiles = selected_molecule[props.index('canonical_smiles')]

            m = Chem.MolFromSmiles(smiles)

            drawer = Draw.rdMolDraw2D.MolDraw2DCairo(500, 125)
            drawer.SetFontSize(1.0)
            drawer.DrawMolecule(m)
            drawer.FinishDrawing()

            img_binary = "data:image/png;base64," + \
                         base64.b64encode(drawer.GetDrawingText()).decode("utf-8")

            td.append(html.Td(html.Img(src=img_binary)))
            td.append(html.Td(smiles, style={'wordWrap': 'break-word'}))
            for key in display_properties:
                if key in PROP_DISP_NAME:
                    td.append(html.Td(selected_molecule[props.index(key)],
                                      style={'text-align': 'center'}))

            molregno = selected_molecule[0]
            if chembl_ids:
                td.append(html.Td(selected_chembl_id))
            else:
                td.append(html.Td(
                    dbc.Button('Highlight',
                               id={'role': 'bt_star_candidate',
                                   'chemblId': selected_chembl_id,
                                   'molregno': str(molregno)
                                   },
                               n_clicks=0)
                ))

            td.append(html.Td(
                dbc.Button('Add',
                           id={'role': 'bt_add_candidate',
                               'chemblId': selected_chembl_id,
                               'molregno': str(molregno)
                               },
                           style={'margin-right': '6px'},
                           n_clicks=0)
            ))

            td.append(html.Td(
                dbc.Button('Analogue',
                        id={'role': 'bt_analoguing_candidate',
                            'chemblId': selected_chembl_id,
                            'molregno': str(molregno),
                            #'smiles': smiles
                            },
                        n_clicks=0)
            ))

            prop_recs.append(html.Tr(td, style={'fontSize': '125%'}))

        return html.Table(prop_recs, style={'width': '100%', 'border': '1px solid lightgray'}), all_props

    def constuct_layout(self):
        # TODO: avoid calling self.cluster_wf.df_embedding
        fig, _ = self.create_graph(self.cluster_wf.df_embedding)

        return html.Div([
            html.Div(className='row', children=[
                dcc.Graph(id='main-figure', figure=fig,
                          className='nine columns',
                          style={'verticalAlign': 'text-top'}),

                html.Div([
                    dcc.Markdown("""**Molecule(s) of Interest**"""),
                    dcc.Markdown(children="""Click *Highlight* to populate this list""",
                                style={'marginTop': 18}),
                    dcc.Markdown("Please enter ChEMBL ID(s) separated by commas."),

                    html.Div(className='row', children=[
                        dcc.Input(id='north_star', type='text', debounce=True, className='nine columns'),
                        dbc.Button('Highlight',
                                   id='bt_north_star', n_clicks=0,
                                   className='three columns'),
                    ], style={'marginLeft': 0, 'marginBottom': 18, }),

                    dcc.Markdown("For fingerprint changes to take effect, first *Apply* the *GPU KMeans-UMAP* Workflow, then *Recluster*"),
                    html.Div(className='row', children=[
                        dcc.Markdown("Fingerprint Radius", style={'marginTop': 12,}),
                        dcc.Input(id='fingerprint_radius', value=2),
                        ], style={'marginLeft': 0, 'marginTop': '6px'}
                    ),

                    html.Div(className='row', children=[
                        dcc.Markdown("Fingerprint Size", style={'marginTop': 12,}),
                        dcc.Input(id='fingerprint_nBits', value=512),
                        ], style={'marginLeft': 0, 'marginTop': '6px'}
                    ),

                    dcc.Tabs([
                        dcc.Tab(label='Cluster Molecules', children=[
                            dcc.Markdown("""**Select Workflow**""", style={'marginTop': 18, }),

                            html.Div(className='row', children=[
                                html.Div(children=[
                                    dcc.Dropdown(id='sl_wf',
                                                 multi=False,
                                                 options=[{'label': 'GPU KMeans-UMAP - Single and Multiple GPUs',
                                                           'value': 'cuchem.wf.cluster.gpukmeansumap.GpuKmeansUmapHybrid'},
                                                          {'label': 'GPU KMeans-UMAP',
                                                           'value': 'cuchem.wf.cluster.gpukmeansumap.GpuKmeansUmap'},
                                                          {'label': 'GPU KMeans-Random Projection - Single GPU',
                                                           'value': 'cuchem.wf.cluster.gpurandomprojection.GpuWorkflowRandomProjection'},
                                                          {'label': 'CPU KMeans-UMAP',
                                                           'value': 'cuchem.wf.cluster.cpukmeansumap.CpuKmeansUmap'}, ],
                                                 value=self.cluster_wf_cls,
                                                 clearable=False),
                                ], className='nine columns'),
                                dbc.Button('Apply',
                                           id='bt_apply_wf', n_clicks=0,
                                           className='three columns'),
                            ], style={'marginLeft': 0, 'marginTop': 6, }),

                            dcc.Markdown("""**Cluster Selection**""", style={'marginTop': 18, }),
                            dcc.Markdown("Set number of clusters", style={'marginTop': 12, }),
                            dcc.Input(id='sl_nclusters', value=self.n_clusters),
                            dcc.Markdown("Click a point to select a cluster.", style={'marginTop': 12, }),

                            html.Div(className='row', children=[
                                dcc.Input(id='selected_clusters', type='text', className='nine columns'),
                                dbc.Button('Recluster',
                                           id='bt_recluster_clusters', n_clicks=0,
                                           className='three columns'),
                            ], style={'marginLeft': 0}),

                            dcc.Markdown("""**Selection Points**""", style={'marginTop': 18, }),
                            dcc.Markdown("""Choose the lasso or rectangle tool in the graph's menu
                                bar and then select points in the graph.
                                """, style={'marginTop': 12, }),
                            dbc.Button('Recluster Selection', id='bt_recluster_points', n_clicks=0),
                            html.Div(children=[html.Div(id='selected_point_cnt'), ]),

                            dbc.Button('Reload', id='bt_reset', n_clicks=0, style={'marginLeft': 0, 'marginTop': 18, }),
                        ]),

                        dcc.Tab(label='Generate Molecules', children=[
                            dcc.Markdown("""**Select Generative Model**""", style={'marginTop': 18, }),

                            html.Div(children=[
                                dcc.Dropdown(id='sl_generative_wf', multi=False,
                                             options=[{'label': 'CDDD Model',
                                                       'value': 'cuchem.wf.generative.Cddd'},
                                                      {'label': 'MegaMolBART Model',
                                                       'value': 'cuchem.wf.generative.MegatronMolBART'},
                                                      ],
                                             value=self.generative_wf_cls,
                                             clearable=False),
                            ]),

                            dcc.RadioItems(
                                id='rd_generation_type',
                                options=[
                                    {'label': 'Interpolate between two molecules', 'value': 'INTERPOLATE'},
                                    {'label': 'Fit cluster to property and extrapolate', 'value': 'EXTRAPOLATE'},
                                    {'label': 'Sample around one molecule', 'value': 'SAMPLE'},
                                ],
                                value='INTERPOLATE',
                                style={'marginTop': 18},
                                inputStyle={'display': 'inline-block', 'marginLeft': 6, 'marginRight': 6},
                                labelStyle={'display': 'block', 'marginLeft': 6, 'marginRight': 6}
                            ),

                            html.Div(className='row', children=[
                                dcc.Markdown("Number to be generated from each compound", 
                                             style={'marginLeft': 10, 'marginTop': 12, 'width': '250px'}),
                                dcc.Input(id='n2generate', value=10),
                            ], style={'marginLeft': 0}),

                            html.Div(className='row', children=[
                                html.Label([
                                    "Select molecular property for fitting and extrapolation",
                                    dcc.Dropdown(id='extrap_compound_property', multi=False,  clearable=False,
                                                options=[{"label": PROP_DISP_NAME[p], "value": p} for p in IMP_PROPS],
                                                value=IMP_PROPS[0]),
                                ], style={'marginTop': 18, 'marginLeft': 18})],
                            ),

                            html.Div(className='row', children=[
                                dcc.Markdown("Cluster number for fitting property and extrapolation", style={'marginLeft': 10, 'marginTop': 12, 'width': '250px'}),
                                dcc.Input(id='extrap_cluster_number', value=0),
                            ], style={'marginLeft': 0}),

                            html.Div(className='row', children=[
                                dcc.Markdown("Step-size for extrapolation", style={'marginLeft': 10, 'marginTop': 12, 'width': '250px'}),
                                dcc.Input(id='extrap_step_size', value=0.1),
                            ], style={'marginLeft': 0}),

                            html.Div(className='row', children=[
                                dcc.Markdown("Number of compounds to extrapolate", style={'marginLeft': 10, 'marginTop': 12, 'width': '250px'}),
                                dcc.Input(id='extrap_n_compounds', value=10),
                            ], style={'marginLeft': 0}),

                            html.Div(className='row', children=[
                                dcc.Markdown("Scaled sampling radius (int, start with 1)",
                                             style={'marginLeft': 10, 'marginTop': 12, 'width': '250px'}),
                                dcc.Input(id='scaled_radius', value=1),
                            ], style={'marginLeft': 0, 'marginTop': '6px'}),

                            dcc.Markdown(children="""**Please Select Two**""",
                                         id="mk_selection_msg",
                                         style={'marginTop': 18}),
                            dcc.Markdown(children="""Click *Add* to populate this list""",
                                         style={'marginTop': 18}),
                            dcc.Checklist(
                                id='ckl_candidate_mol_id',
                                options=[],
                                value=[],
                                inputStyle={'display': 'inline-block', 'marginLeft': 6, 'marginRight': 6},
                                labelStyle={'display': 'block', 'marginLeft': 6, 'marginRight': 6}
                            ),
                            html.Div(className='row', children=[
                                dbc.Button('Generate', id='bt_generate', n_clicks=0, style={'marginRight': 12}),
                                dbc.Button('Reset', id='bt_reset_candidates', n_clicks=0),
                            ], style={'marginLeft': 0}),
                        ]),

                        dcc.Tab(label='Predict Properties', children=[

                            dcc.Markdown("""**Select Featurizing Model**""", style={'marginTop': 18,}),
                            html.Div(children=[
                                dcc.Dropdown(id='sl_featurizing_wf', multi=False,
                                             options=[{'label': 'CDDD Model',
                                                       'value': 'cuchem.wf.generative.Cddd'},
                                                       {'label': 'MolBART Model',
                                                       'value': 'cuchem.wf.generative.MolBART'},
                                                       {'label': 'MegatronMolBART Model',
                                                       'value': 'cuchem.wf.generative.MegatronMolBART'},
                                                     ],
                                             value=self.generative_wf_cls,
                                             clearable=False),
                            ]),
                             html.Div(className='row', children=[
                                html.Label([
                                    "Select molecular property for fitting and prediction",
                                    dcc.Dropdown(id='fit_nn_compound_property', multi=False,  clearable=False,
                                    options=[{"label": PROP_DISP_NAME[p], "value": p} for p in IMP_PROPS],
                                    value=IMP_PROPS[0]),
                                ], style={'marginTop': 18, 'marginLeft': 18})],
                            ),
                            html.Div(className='row', children=[
                                dcc.Markdown("Train cluster", style={'marginTop': 12,}),
                                dcc.Input(id='fit_nn_train_cluster_number', value=0),
                                ], style={'marginLeft': 0, 'marginTop': '6px'}
                            ),
                            html.Div(className='row', children=[
                                dcc.Markdown("Test cluster", style={'marginTop': 12,}),
                                dcc.Input(id='fit_nn_test_cluster_number', value=1),
                                ], style={'marginLeft': 0, 'marginTop': '6px'}
                            ),
                            dcc.Markdown(children="""**Neural Network Parameters**""",
                                         id="nn_params_msg",
                                         style={'marginTop': 18}
                            ),
                            html.Div(className='row', children=[
                                dcc.Markdown("Hidden layer sizes", style={'marginTop': 12,}),
                                dcc.Input(id='fit_nn_hidden_layer_sizes', value=''),
                                ], style={'marginLeft': 0, 'marginTop': '6px'}
                            ),
                            html.Div(className='row', children=[
                                dcc.Markdown("Activation Function", style={'marginTop': 12,}),
                                dcc.Input(id='fit_nn_activation_fn', value='LeakyReLU'),
                                ], style={'marginLeft': 0, 'marginTop': '6px'}
                            ),
                            html.Div(className='row', children=[
                                dcc.Markdown("Final Activation Function", style={'marginTop': 12,}),
                                dcc.Input(id='fit_nn_final_activation_fn', value='LeakyReLU'),
                                ], style={'marginLeft': 0, 'marginTop': '6px'}
                            ),
                            html.Div(className='row', children=[
                                dcc.Markdown("Number of training epochs", style={'marginTop': 12,}),
                                dcc.Input(id='fit_nn_max_epochs', value=10),
                                ], style={'marginLeft': 0, 'marginTop': '6px'}
                            ),
                            html.Div(className='row', children=[
                                dcc.Markdown("Learning Rate", style={'marginTop': 12,}),
                                dcc.Input(id='fit_nn_learning_rate', value=0.001),
                                ], style={'marginLeft': 0, 'marginTop': '6px'}
                            ),
                            html.Div(className='row', children=[
                                dcc.Markdown("Weight Decay (Adam)", style={'marginTop': 12,}),
                                dcc.Input(id='fit_nn_weight_decay', value=0.0001),
                                ], style={'marginLeft': 0, 'marginTop': '6px'}
                            ),
                            html.Div(className='row', children=[
                                dcc.Markdown("Batch size", style={'marginTop': 12,}),
                                dcc.Input(id='fit_nn_batch_size', value=1),
                                ], style={'marginLeft': 0, 'marginTop': '6px'}
                            ),
                            html.Div(className='row', children=[
                                dbc.Button('Fit', id='bt_fit', n_clicks=0, style={'marginRight': 12}),
                                ], style={'marginLeft': 0}
                            ),
                        ]),

                        dcc.Tab(label='Find Analogues', children=[

                            html.Div(className='row', children=[
                                dcc.Markdown("Maxinum Number of Analogues", style={'marginTop': 12,}),
                                dcc.Input(id='analoguing_n_analogues', value=10),
                                ], style={'marginLeft': 0, 'marginTop': '6px'}
                            ),
                            html.Div(className='row', children=[
                                dcc.Markdown("Similarity Threshold", style={'marginTop': 12,}),
                                dcc.Input(id='analoguing_threshold', value=0.33),
                                ], style={'marginLeft': 0, 'marginTop': '6px'}
                            ),
                            html.Div(children=[
                                dcc.Dropdown(id='analoguing_type', multi=False,
                                             options=[{'label': 'Similar compounds',
                                                       'value': 'similar'},
                                                       {'label': 'Compounds with the same scaffold',
                                                       'value': 'scaffold'},
                                                       {'label': 'Compounds that are superstructures',
                                                       'value': 'superstructure'},
                                                     ],
                                             value='similar',
                                             clearable=False),
                            ]),
                            dcc.Markdown(children="""Choose a compound""",
                                         id="analoguing_msg",
                                         style={'marginTop': 18}
                            ),
                            dcc.Markdown(children="""Click *Analogue* to populate this list""",
                                         style={'marginTop': 18}),
                            dcc.Checklist(
                                id='ckl_analoguing_mol_id',
                                options=[],
                                value=[],
                                #inputStyle={'display': 'inline-block', 'marginLeft': 6, 'marginRight': 6},
                                #labelStyle={'display': 'block', 'marginLeft': 6, 'marginRight': 6}
                            ),
                            html.Div(className='row', children=[
                                dbc.Button('Search', id='bt_analoguing', n_clicks=0, style={'marginRight': 12}),
                            ], style={'marginLeft': 0}),
                        ])

                    ]),

                    html.Div(className='row', children=[
                        html.Label([
                            "Select molecular property for color gradient",
                            dcc.Dropdown(id='sl_prop_gradient', multi=False, clearable=True,
                                         options=[{"label": PROP_DISP_NAME[p], "value": p} for p in IMP_PROPS], ),
                        ], style={'marginTop': 18, 'marginLeft': 18})],
                             ),
                ], className='three columns', style={'marginLeft': 18, 'marginTop': 90, 'verticalAlign': 'text-top', }),
            ]),

            html.Div(id='section_generated_molecules', 
                children=[
                    html.A(
                        'Export to SDF',
                        id='download-link',
                        download="rawdata.sdf",
                        href="/cheminfo/downloadSDF",
                        target="_blank",
                        n_clicks=0,
                        style={'marginLeft': 10, 'fontSize': '150%'}
                    ),
                    html.Div(id='table_generated_molecules', children=[]),
                ], 
                style={'display': 'none'}
            ),

            html.Div(id='section_generated_molecules_clustered', children=[
                dcc.Graph(id='gen_figure', figure=fig,
                        #className='nine columns',
                        #style={'verticalAlign': 'text-top'}
                ),
            ], style={'display': 'none'}),

            html.Div(id='section_fitting', children=[
                dcc.Graph(id='fitting_figure', figure=fig,
                        #className='nine columns',
                        #style={'verticalAlign': 'text-top'}
                ),
            ], style={'display': 'none'}),

            html.Div(id='section_selected_molecules', children=[
                html.Div(className='row', children=[
                    html.Div(id='section_display_properties', children=[
                        html.Label([
                            "Select Molecular Properties",
                            dcc.Dropdown(id='sl_mol_props', multi=True,
                                        options=[
                                            {'label': 'alogp', 'value': 'alogp'}],
                                        value=['alogp']),
                        ])],
                        className='nine columns'),
                    html.Div(children=[
                        dbc.Button("<", id="bt_page_prev",
                                style={"height": "25px"}),
                        html.Span(children=1, id='current_page',
                                style={"paddingLeft": "6px"}),
                        html.Span(children=' of 1', id='total_page',
                                style={"paddingRight": "6px"}),
                        dbc.Button(">", id="bt_page_next",
                                style={"height": "25px"})
                        ],
                        className='three columns',
                        style={'verticalAlign': 'text-bottom', 'text-align': 'right'}
                    ),
                ], style={'margin': 12}),

                html.Div(
                    id='tb_selected_molecules', 
                    children=[], 
                    style={'width': '100%'}
                )
            ], style={'display': 'none', 'width': '100%'}),

            html.Div(id='section_analoguing', children=[
                html.Div(children=[
                    html.Div(id='tb_analoguing', children=[],
                             style={'verticalAlign': 'text-top'}
                             ),
                ])
            ], style={'display': 'none'}),

            html.Div(id='refresh_main_fig', style={'display': 'none'}),
            html.Div(id='northstar_cluster', style={'display': 'none'}),
            html.Div(id='recluster_error', style={'display': 'none'}),
            html.Div(id='mol_selection_error', style={'display': 'none'}),
            html.Div(id='show_selected_mol', style={'display': 'none'}),
            html.Div(id='show_generated_mol', style={'display': 'none'}),
            html.Div(id='generation_candidates', style={'display': 'none'}),
            html.Div(id='refresh_moi_prop_table', style={'display': 'none'}),
            html.Div(id='interpolation_error', style={'display': 'none'}),
            html.Div(id='analoguing_candidates', style={'display': 'none'}), # Not displayed but used to keep track of compounds added to checklist of compounds to be analogued

            html.Div(className='row', children=[
                dbc.Modal([
                    dbc.ModalHeader("Error"),
                    dbc.ModalBody(
                        html.Div(id='error_msg', style={'color': 'red'}),
                    ),
                    dbc.ModalFooter(
                        dbc.Button("Close", id="bt_close_err", className="ml-auto")
                    ),
                ], id="md_error"),
            ]),
        ])

    def handle_error(self, recluster_error, interpolation_error, bt_close_err):
        comp_id, event_type = self._fetch_event_data()

        if comp_id == 'bt_close_err' and event_type == 'n_clicks':
            return '', False

        msg = None
        if comp_id == 'interpolation_error' and event_type == 'children':
            msg = interpolation_error
        elif comp_id == 'recluster_error' and event_type == 'children':
            msg = recluster_error

        if msg is None:
            raise dash.exceptions.PreventUpdate
        return msg, True

    @report_ui_error(6)
    def handle_molecule_selection(self, mf_selected_data, selected_columns,
                                  prev_click, next_click, refresh_moi_prop_table,
                                  north_star, current_page, show_selected_mol,
                                  sl_prop_gradient):
        comp_id, event_type = self._fetch_event_data()

        module_details = None
        chembl_ids = None
        # Code to support pagination
        if comp_id == 'bt_page_prev' and event_type == 'n_clicks':
            if current_page == 1:
                raise dash.exceptions.PreventUpdate
            current_page -= 1
        elif comp_id == 'bt_page_next' and event_type == 'n_clicks':
            if len(mf_selected_data['points']) < PAGE_SIZE * (current_page + 1):
                raise dash.exceptions.PreventUpdate
            current_page += 1
        elif north_star and \
                ((comp_id == 'refresh_moi_prop_table' and event_type == 'children')):
            chembl_ids = north_star.split(",")
        elif (comp_id == 'main-figure' and event_type == 'selectedData') or \
                (comp_id == 'sl_mol_props' and event_type == 'value'):
            pass
        else:
            raise dash.exceptions.PreventUpdate

        if selected_columns and sl_prop_gradient:
            if sl_prop_gradient not in selected_columns:
                selected_columns.append(sl_prop_gradient)

        module_details, all_props = self.construct_molecule_detail(
            mf_selected_data, selected_columns, current_page,
            pageSize=PAGE_SIZE, chembl_ids=chembl_ids)

        if module_details is None and all_props is None:
            return dash.no_update, dash.no_update, dash.no_update, \
                   dash.no_update, dash.no_update, dash.no_update,

        if chembl_ids:
            last_page = ''
        else:
            last_page = ' of ' + str(len(mf_selected_data['points']) // PAGE_SIZE)

        if show_selected_mol is None:
            show_selected_mol = 0
        show_selected_mol += 1

        return module_details, all_props, current_page, last_page, show_selected_mol, dash.no_update

    def handle_data_selection(self, mf_click_data, mf_selected_data,
                              bt_cluster_clicks, bt_point_clicks,
                              northstar_cluster,
                              curr_clusters):
        comp_id, event_type = self._fetch_event_data()
        selected_clusters = ''
        selected_point_cnt = ''

        if comp_id == 'main-figure' and event_type == 'clickData':
            # Event - On selecting cluster on the main scatter plot
            clusters = []
            if curr_clusters:
                clusters = list(map(int, curr_clusters.split(",")))

            points = mf_click_data['points']
            for point in points:
                cluster = point['text']
                if cluster in clusters:
                    clusters.remove(cluster)
                else:
                    clusters.append(cluster)
            selected_clusters = ','.join(map(str, clusters))

        elif comp_id == 'main-figure' and event_type == 'selectedData':
            # Event - On selection on the main scatterplot
            if not mf_selected_data:
                raise dash.exceptions.PreventUpdate

            points = mf_selected_data['points']
            selected_point_cnt = str(len(points)) + ' points selected'
            clusters = {point['text'] for point in points}
            selected_clusters = northstar_cluster

        elif comp_id == 'northstar_cluster' and event_type == 'children':
            selected_clusters = northstar_cluster
        elif (comp_id == 'bt_recluster_clusters' and event_type == 'n_clicks') \
                or (comp_id == 'bt_recluster_points' and event_type == 'n_clicks'):
            selected_clusters = northstar_cluster
        else:
            raise dash.exceptions.PreventUpdate

        return selected_clusters, selected_point_cnt

    def handle_mark_north_star(self, bt_north_star_click, north_star):
        comp_id, event_type = self._fetch_event_data()

        if event_type != 'n_clicks' or dash.callback_context.triggered[0]['value'] == 0:
            raise dash.exceptions.PreventUpdate

        selected_north_star = []
        selected_north_star_mol_reg_id = []

        if north_star:
            selected_north_star = north_star.split(",")
            selected_north_star_mol_reg_id = [
                str(row[0]) for row in self.chem_data.fetch_molregno_by_chemblId(selected_north_star)]

        comp_detail = json.loads(comp_id)
        selected_chembl_id = comp_detail['chemblId']

        if selected_chembl_id not in selected_north_star:
            selected_north_star.append(selected_chembl_id)
            selected_north_star_mol_reg_id.append(comp_detail['molregno'])
        return ','.join(selected_north_star)

    @report_ui_error(4)
    def handle_re_cluster(
        self, bt_cluster_clicks, bt_point_clicks, bt_north_star_clicks,
        sl_prop_gradient, sl_nclusters, refresh_main_fig,
        fingerprint_radius, fingerprint_nBits,
        selected_clusters, selected_points, north_star, refresh_moi_prop_table
    ):
        comp_id, event_type = self._fetch_event_data()

        if comp_id == 'sl_nclusters':
            if sl_nclusters:
                self.n_clusters = int(sl_nclusters)
                self.cluster_colors = generate_colors(self.n_clusters)

            raise dash.exceptions.PreventUpdate

        if comp_id in ['fingerprint_radius', 'fingerprint_nBits']:
            raise dash.exceptions.PreventUpdate

        filter_values = None
        filter_column = None
        reload_data = False
        recluster_data = True
        moi_molregno = None
        _refresh_moi_prop_table = dash.no_update

        if comp_id == 'bt_recluster_clusters' and event_type == 'n_clicks':
            if selected_clusters:
                filter_values = list(map(int, selected_clusters.split(",")))
            filter_column = 'cluster'
        elif selected_points and comp_id == 'bt_recluster_points' and event_type == 'n_clicks':
            filter_values = []
            for point in selected_points['points']:
                if 'customdata' in point:
                    filter_values.append(point['customdata'])
            filter_column = 'id'
        elif comp_id == 'bt_north_star' and event_type == 'n_clicks':
            if north_star:
                north_star = north_star.split(',')
                missing_mols, molregnos, _ = self.cluster_wf.add_molecules(
                    north_star, radius=int(fingerprint_radius), nBits=int(fingerprint_nBits))
                recluster_data = len(missing_mols) > 0
                logger.info("%d missing molecules added...", len(missing_mols))
                logger.debug("Missing molecules were %s", missing_mols)

                moi_molregno = " ,".join(list(map(str, molregnos)))
                if refresh_moi_prop_table is None:
                    refresh_moi_prop_table = 0
                _refresh_moi_prop_table = refresh_moi_prop_table + 1
            else:
                raise dash.exceptions.PreventUpdate

        elif comp_id == 'refresh_main_fig' and event_type == 'children':
            reload_data = True
            recluster_data = True
        else:
            # Event that are expected to reach this block are
            #   'sl_prop_gradient' and event_type == 'value':
            reload_data = False
            recluster_data = False

        if north_star and moi_molregno is None:
            molregnos = [row[0] for row in self.cluster_wf.dao.fetch_id_from_chembl(north_star.split(','))]
            moi_molregno = " ,".join(list(map(str, molregnos)))

        figure, northstar_cluster = self.recluster_selection(
            filter_value=filter_values,
            filter_column=filter_column,
            gradient_prop=sl_prop_gradient,
            north_stars=moi_molregno,
            color_col='cluster',
            reload_data=reload_data,
            recluster_data=recluster_data, 
            fingerprint_radius=int(fingerprint_radius), 
            fingerprint_nBits=int(fingerprint_nBits)
        )

        return figure, ','.join(northstar_cluster), _refresh_moi_prop_table, dash.no_update
