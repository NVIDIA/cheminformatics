# Copyright 2020 NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import base64
import logging
from pydoc import locate
from io import StringIO

import flask
from flask import send_file, Response

from rdkit import Chem
from rdkit.Chem import Draw, PandasTools

import pandas as pd

import cupy
import plotly.graph_objects as go
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State, ALL

from nvidia.cheminformatics.utils import generate_colors, report_ui_error
from nvidia.cheminformatics.data.helper.chembldata import ChEmblData, IMP_PROPS
from nvidia.cheminformatics.decorator import LipinskiRuleOfFiveDecorator
from nvidia.cheminformatics.decorator import MolecularStructureDecorator
from nvidia.cheminformatics.utils.singleton import Singleton

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
    for row, data in vis.genreated_df.iterrows():
        mol = Chem.MolFromSmiles(data['SMILES'])
        if (mol is not None):
            valid_idx.append(row)

    valid_df = vis.genreated_df.iloc[valid_idx]
    valid_df = valid_df[col_list]

    PandasTools.AddMoleculeColumnToFrame(valid_df,'SMILES')
    PandasTools.WriteSDF(valid_df, output, properties=list(valid_df.columns))

    output.seek(0)

    return Response(
        output.getvalue(),
        mimetype="text/application",
        headers={"Content-disposition":
                 "attachment; filename=download.sdf"})


class ChemVisualization(metaclass=Singleton):

    def __init__(self, cluster_wf):
        self.app = app
        self.cluster_wf = cluster_wf
        self.n_clusters = cluster_wf.n_clusters
        self.chem_data = ChEmblData()
        self.genreated_df = None
        self.cluster_wf_cls = 'nvidia.cheminformatics.wf.cluster.gpukmeansumap.GpuKmeansUmap'
        self.generative_wf_cls = 'nvidia.cheminformatics.wf.generative.Cddd'

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
             Input('refresh_main_fig', 'children') ],
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
            Output('genration_candidates', 'children'),
            [Input({'role': 'bt_add_candidate', 'chemblId': ALL, 'molregno': ALL}, 'n_clicks'),
             Input('bt_reset_candidates', 'n_clicks'),],
            State('genration_candidates', 'children'))(self.handle_add_candidate)

        self.app.callback(
            Output('ckl_candidate_mol_id', 'options'),
            Input('genration_candidates', 'children'))(self.handle_construct_candidates)

        self.app.callback(
            [Output('ckl_candidate_mol_id', 'value'),
             Output('mk_selection_msg', 'children')],
            [Input('ckl_candidate_mol_id', 'value'),
             Input('rd_generation_type', 'value')])(self.handle_ckl_selection)

        self.app.callback(
            [Output('table_generated_molecules', 'children'),
             Output('show_generated_mol', 'children'),
             Output('interpolation_error', 'children'),],
            [Input("bt_generate", "n_clicks"),],
            [State('sl_generative_wf', 'value'),
             State('ckl_candidate_mol_id', 'value'),
             State('n2generate', 'value'),
             State('scaled_radius', 'value'),
             State('rd_generation_type', 'value'),
             State('show_generated_mol', 'children')])(self.handle_generation)

        self.app.callback(
            [Output('section_generated_molecules', 'style'),
             Output('section_selected_molecules', 'style'),],
            [Input('show_generated_mol', 'children'),
             Input('show_selected_mol', 'children')])(self.handle_property_tables)


    def handle_add_candidate(self, bt_add_candidate,
                                    bt_reset_candidates,
                                    genration_candidates):
        comp_id, event_type = self._fetch_event_data()

        if comp_id == 'bt_reset_candidates' and event_type == 'n_clicks':
            return ''

        if event_type != 'n_clicks' or dash.callback_context.triggered[0]['value'] == 0:
            raise dash.exceptions.PreventUpdate

        selected_candidates = []

        if genration_candidates:
            selected_candidates = genration_candidates.split(",")

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
            return {'display': 'none'}, {'display': 'block'}
        elif comp_id == 'show_generated_mol' and event_type == 'children':
            return {'display': 'block'}, {'display': 'none'}
        return dash.no_update, dash.no_update

    @report_ui_error(3)
    def handle_generation(self, bt_generate,
                      sl_generative_wf, ckl_candidate_mol_id,
                      n2generate, scaled_radius, rd_generation_type, show_generated_mol):
        comp_id, event_type = self._fetch_event_data()

        chemble_ids = []
        if comp_id == 'bt_generate' and event_type == 'n_clicks':
            chemble_ids = ckl_candidate_mol_id
        else:
            return dash.no_update, dash.no_update

        self.generative_wf_cls = sl_generative_wf
        wf_class = locate(self.generative_wf_cls)
        generative_wf = wf_class()
        n2generate = int(n2generate)
        scaled_radius = int(scaled_radius)

        if rd_generation_type == 'SAMPLE':
            self.genreated_df = generative_wf.find_similars_smiles_from_id(chemble_ids,
                                                                           num_requested=n2generate,
                                                                           scaled_radius=scaled_radius,
                                                                           force_unique=True)
        else:
            self.genreated_df = generative_wf.interpolate_from_id(chemble_ids,
                                                             num_points=n2generate,
                                                             scaled_radius=scaled_radius,
                                                             force_unique=True)

        if show_generated_mol is None:
            show_generated_mol = 0
        show_generated_mol += 1

        # Add other useful attributes to be added for rendering
        self.genreated_df = MolecularStructureDecorator().decorate(self.genreated_df)
        self.genreated_df = LipinskiRuleOfFiveDecorator().decorate(self.genreated_df)

        # Create Table header
        table_headers = []
        columns = self.genreated_df.columns.to_list()
        for column in columns:
            table_headers.append(html.Th(column, style={'fontSize': '150%', 'text-align': 'center'}))

        prop_recs = [html.Tr(table_headers, style={'background': 'lightgray'})]
        for row_idx in range(self.genreated_df.shape[0]):
            td = []

            try:
                col_pos = columns.index('Chemical Structure')
                col_data = self.genreated_df.iat[row_idx, col_pos]

                if 'value' in col_data and col_data['value'] == 'Error interpreting SMILES using RDKit':
                    continue
            except ValueError:
                pass

            for col_id in range(len(columns)):
                col_data = self.genreated_df.iat[row_idx, col_id]

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

        return html.Table(prop_recs, style={'width': '100%', 'margin': 12, 'border': '1px solid lightgray'}), show_generated_mol, dash.no_update

    def handle_ckl_selection(self, ckl_candidate_mol_id, rd_generation_type):
        selection_msg = '**Please Select Two Molecules**'
        selection_cnt = 2

        if rd_generation_type == 'SAMPLE':
            selection_msg = '**Please Select One Molecule**'
            selection_cnt = 1

        if ckl_candidate_mol_id and len(ckl_candidate_mol_id) > selection_cnt:
            ckl_candidate_mol_id = ckl_candidate_mol_id[selection_cnt * -1:]

        return ckl_candidate_mol_id, selection_msg

    def handle_construct_candidates(self, north_star):
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
            return self.cluster_wf.cluster()
        else:
            return self.cluster_wf.recluster(filter_column, filter_values,
                                            n_clusters=self.n_clusters)

    def recluster_selection(self,
                           filter_value=None,
                           filter_column=None,
                           gradient_prop=None,
                           north_stars=None,
                           reload_data=False,
                           recluster_data=True,
                           color_col='cluster'):

        if recluster_data or self.cluster_wf.df_embedding is None:
            df_embedding = self.recluster(filter_values=filter_value,
                                          filter_column=filter_column,
                                          reload_data=reload_data)
        else:
            df_embedding = self.cluster_wf.df_embedding

        return self.create_graph(df_embedding,
                                 color_col=color_col,
                                 gradient_prop=gradient_prop,
                                 north_stars=north_stars)

    def create_graph(self, ldf, color_col='cluster', north_stars=None, gradient_prop=None):
        fig = go.Figure(layout={'colorscale': {}})

        # Filter out relevant columns in this method.
        if hasattr(ldf, 'compute'):
            relevant_cols = ['id', 'x', 'y', 'cluster']
            if gradient_prop:
                relevant_cols.append(gradient_prop)
            if color_col is not 'cluster':
                relevant_cols.append(color_col)

            ldf = ldf.iloc[:, ldf.columns.isin(relevant_cols)]
            ldf = ldf.compute()

        moi_molregno = []
        if north_stars:
            moi_molregno = north_stars.split(",")

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
                        'text': cluster,
                        'customdata': customdata,
                        'name': 'Cluster ' + str(cluster_id),
                        'mode': 'markers',
                        'marker': {
                            'size': df_size,
                            'symbol': df_shape,
                            'color': self.cluster_colors[int(cluster_id) % len(self.cluster_colors)],
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
        table_headers = [html.Th("Chemical Structure", style={'width': '30%', 'fontSize': '150%', 'text-align': 'center'}),
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
            props, selected_molecules = self.chem_data.fetch_props_by_chemble(chembl_ids)
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
            td.append(html.Td(smiles, style={'maxWidth': '100px', 'wordWrap':'break-word'}))
            for key in display_properties:
                if key in PROP_DISP_NAME:
                    td.append(html.Td(selected_molecule[props.index(key)]))

            molregno = selected_molecule[0]
            if chembl_ids:
                td.append(html.Td(selected_chembl_id))
            else:
                td.append(html.Td(
                    dbc.Button('Add as MoI',
                            id={'role': 'bt_star_candidate',
                                'chemblId': selected_chembl_id,
                                'molregno': str(molregno)
                                },
                            n_clicks=0)
                ))

            td.append(html.Td(
                dbc.Button('Add for Interpolation',
                        id={'role': 'bt_add_candidate',
                            'chemblId': selected_chembl_id,
                            'molregno': str(molregno)
                            },
                        n_clicks=0)
            ))

            prop_recs.append(html.Tr(td))

        return html.Table(prop_recs, style={'width': '100%', 'margin': 12, 'border': '1px solid lightgray'}), all_props

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
                    dcc.Markdown("Please enter ChEMBL ID(s) separated by commas."),

                    html.Div(className='row', children=[
                        dcc.Input(id='north_star', type='text', debounce=True, className='nine columns'),
                        dbc.Button('Highlight',
                                   id='bt_north_star', n_clicks=0,
                                   className='three columns'),
                    ], style={'marginLeft': 0, 'marginBottom': 18,}),

                    dcc.Tabs([
                        dcc.Tab(label='Cluster Molecules', children=[
                            dcc.Markdown("""**Select Workflow**""", style={'marginTop': 18,}),

                            html.Div(className='row', children=[
                                html.Div(children=[
                                    dcc.Dropdown(id='sl_wf',
                                                 multi=False,
                                                 options=[{'label': 'GPU KMeans-UMAP', 'value': 'nvidia.cheminformatics.wf.cluster.gpukmeansumap.GpuKmeansUmap'},
                                                          {'label': 'GPU KMeans-UMAP - Single and Multiple GPUs', 'value': 'nvidia.cheminformatics.wf.cluster.gpukmeansumap.GpuKmeansUmapHybrid'},
                                                          {'label': 'GPU KMeans-Random Projection - Single GPU', 'value': 'nvidia.cheminformatics.wf.cluster.gpurandomprojection.GpuWorkflowRandomProjection'},
                                                          {'label': 'CPU KMeans-UMAP', 'value': 'nvidia.cheminformatics.wf.cluster.cpukmeansumap.CpuKmeansUmap'},],
                                                 value=self.cluster_wf_cls,
                                                 clearable=False),
                                ], className='nine columns'),
                                dbc.Button('Apply',
                                        id='bt_apply_wf', n_clicks=0,
                                        className='three columns'),
                            ], style={'marginLeft': 0, 'marginTop': 6, }),

                            dcc.Markdown("""**Cluster Selection**""", style={'marginTop': 18,}),
                            dcc.Markdown("Set number of clusters", style={'marginTop': 12,}),
                            dcc.Input(id='sl_nclusters', value=self.n_clusters),
                            dcc.Markdown("Click a point to select a cluster.", style={'marginTop': 12,}),

                            html.Div(className='row', children=[
                                dcc.Input(id='selected_clusters', type='text', className='nine columns'),
                                dbc.Button('Recluster',
                                        id='bt_recluster_clusters', n_clicks=0,
                                        className='three columns'),
                            ], style={'marginLeft': 0}),

                            dcc.Markdown("""**Selection Points**""", style={'marginTop': 18,}),
                            dcc.Markdown("""Choose the lasso or rectangle tool in the graph's menu
                                bar and then select points in the graph.
                                """, style={'marginTop': 12,}),
                            dbc.Button('Recluster Selection', id='bt_recluster_points', n_clicks=0),
                            html.Div(children=[html.Div(id='selected_point_cnt'), ]),

                            dbc.Button('Reload', id='bt_reset', n_clicks=0, style={'marginLeft': 0, 'marginTop': 18, }),
                        ]),

                        dcc.Tab(label='Generate Molecules', children=[
                            dcc.Markdown("""**Select Generative Model**""", style={'marginTop': 18,}),

                            html.Div(children=[
                                dcc.Dropdown(id='sl_generative_wf', multi=False,
                                             options=[{'label': 'CDDD Model',
                                                       'value': 'nvidia.cheminformatics.wf.generative.Cddd'},
                                                     ],
                                             value=self.generative_wf_cls,
                                             clearable=False),
                            ]),

                            dcc.RadioItems(
                                id='rd_generation_type',
                                options=[
                                    {'label': 'Interpolate between two molecules', 'value': 'INTERPOLATE'},
                                    {'label': 'Sample around one molecule', 'value': 'SAMPLE'},
                                ],
                                value='INTERPOLATE',
                                style={'marginTop': 18},
                                inputStyle={'display': 'inline-block', 'marginLeft': 6, 'marginRight': 6},
                                labelStyle={'display': 'block', 'marginLeft': 6, 'marginRight': 6}
                            ),

                            html.Div(className='row', children=[
                                dcc.Markdown("Number of molecules to generate", style={'marginLeft': 10, 'marginTop': 12, 'width': '250px'}),
                                dcc.Input(id='n2generate', value=10),
                            ], style={'marginLeft': 0}),

                            html.Div(className='row', children=[
                                dcc.Markdown("Scaled sampling radius (int, start with 1)", style={'marginLeft': 10, 'marginTop': 12, 'width': '250px'}),
                                dcc.Input(id='scaled_radius', value=1),
                            ], style={'marginLeft': 0, 'marginTop': '6px'}),

                            dcc.Markdown(children="""**Please Select Two**""",
                                         id="mk_selection_msg",
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
                    ]),

                    html.Div(className='row', children=[
                        html.Label([
                            "Select molecular property for color gradient",
                            dcc.Dropdown(id='sl_prop_gradient', multi=False,  clearable=True,
                                        options=[{"label": PROP_DISP_NAME[p], "value": p} for p in IMP_PROPS],),
                        ], style={'marginTop': 18, 'marginLeft': 18})],
                    ),
                ], className='three columns', style={'marginLeft': 18, 'marginTop': 90, 'verticalAlign': 'text-top', }),
            ]),

            html.Div(id='section_generated_molecules', children=[
                 html.A(
                    'Export to SDF',
                    id='download-link',
                    download="rawdata.sdf",
                    href="/cheminfo/downloadSDF",
                    target="_blank",
                    n_clicks=0,
                    style={'marginLeft': 10, 'fontSize': '150%'}
                ),
                html.Div(id='table_generated_molecules', children=[
                ])
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
                        ], style={'marginLeft': 30})],
                        className='nine columns',
                    ),
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
                ]),

                html.Div(children=[
                    html.Div(id='tb_selected_molecules', children=[],
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
            html.Div(id='genration_candidates', style={'display': 'none'}),
            html.Div(id='refresh_moi_prop_table', style={'display': 'none'}),
            html.Div(id='interpolation_error', style={'display': 'none'}),

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
            (comp_id == 'sl_mol_props' and event_type == 'value') :
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
            last_page = ' of ' + str(len(mf_selected_data['points'])//PAGE_SIZE)

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
    def handle_re_cluster(self, bt_cluster_clicks, bt_point_clicks, bt_north_star_clicks,
                          sl_prop_gradient, sl_nclusters, refresh_main_fig,
                          selected_clusters, selected_points, north_star, refresh_moi_prop_table):
        comp_id, event_type = self._fetch_event_data()

        if comp_id == 'sl_nclusters':
            if sl_nclusters:
                self.n_clusters = int(sl_nclusters)
                self.cluster_colors = generate_colors(self.n_clusters)

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
                missing_mols, molregnos, _ = self.cluster_wf.add_molecules(north_star)
                recluster_data = len(missing_mols) > 0
                logger.info("%d missing molecules added...", len(missing_mols))
                logger.debug("Missing molecules werew %s", missing_mols)

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
            reload_data = reload_data,
            recluster_data=recluster_data)

        return figure, ','.join(northstar_cluster), _refresh_moi_prop_table, dash.no_update
