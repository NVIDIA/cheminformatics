# Copyright 2020 NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import base64
import logging
from rdkit import Chem
from rdkit.Chem import Draw
import numpy as np

import cupy
import plotly.graph_objects as go
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State, ALL

from nvidia.cheminformatics.utils import generate_colors, report_ui_error
from nvidia.cheminformatics.data.helper.chembldata import ChEmblData, IMP_PROPS
from nvidia.cheminformatics.fingerprint import MorganFingerprint


logger = logging.getLogger(__name__)

external_stylesheets = [
    'https://codepen.io/chriddyp/pen/bWLwgP.css', dbc.themes.BOOTSTRAP]

main_fig_height = 700
CHEMBL_DB = '/data/db/chembl_27.db'
PAGE_SIZE = 10
DOT_SIZE = 5


class ChemVisualization:

    def __init__(self, workflow, gpu=True):
        self.enable_gpu = gpu
        self.app = dash.Dash(
            __name__, external_stylesheets=external_stylesheets)

        self.workflow = workflow
        self.n_clusters = workflow.n_clusters

        self.chem_data = ChEmblData()

        # Store colors to avoid plots changes colors on events such as
        # molecule selection, etc.
        self.cluster_colors = None

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
             Output('north_star_clusterid_map', 'children'),
             Output('recluster_error', 'children')],
            [Input('bt_recluster_clusters', 'n_clicks'),
             Input('bt_recluster_points', 'n_clicks'),
             Input('bt_north_star', 'n_clicks'),
             Input('hidden_northstar', 'value'),
             Input('sl_prop_gradient', 'value'),
             Input('sl_nclusters', 'value'),
             Input('refresh_main_fig', 'value') ],
            [State("selected_clusters", "value"),
             State("main-figure", "selectedData"),
             State('north_star', 'value'), ])(self.handle_re_cluster)

        # Register callbacks for selection inside main figure to update module details
        self.app.callback(
            [Output('tb_selected_molecules', 'children'),
             Output('sl_mol_props', 'options'),
             Output('current_page', 'children'),
             Output('total_page', 'children'),
             Output('section_molecule_details', 'style'),
             Output('mol_selection_error', 'children')],
            [Input('main-figure', 'selectedData'),
             Input('sl_mol_props', 'value'),
             Input('sl_prop_gradient', 'value'),
             Input('bt_page_prev', 'n_clicks'),
             Input('bt_page_next', 'n_clicks'),
             Input('north_star_clusterid_map', 'children')],
            State('current_page', 'children'))(self.handle_molecule_selection)

        self.app.callback(
            Output("refresh_main_fig", "value"),
            [Input("bt_reset", "n_clicks")],
            [State("refresh_main_fig", "children")])(self.handle_reset)

        self.app.callback(
            [Output('north_star', 'value'),
             Output('hidden_northstar', 'value')],
            [Input({'role': 'bt_star_candidate', 'chemblId': ALL, 'molregno': ALL}, 'n_clicks')],
            [State('north_star', 'value'),
             State('hidden_northstar', 'value')])(self.handle_mark_north_star)

    def _fetch_event_data(self):
        if not dash.callback_context.triggered:
            raise dash.exceptions.PreventUpdate
        return dash.callback_context.triggered[0]['prop_id'].split('.')

    def handle_reset(self, reset_bt_evt, refresh_main_fig):
        if not dash.callback_context.triggered:
            raise dash.exceptions.PreventUpdate

        self.recluster(reload_data=True)
        if refresh_main_fig is None:
            refresh_main_fig = 1
        else:
            refresh_main_fig = int(refresh_main_fig)

        # Change the refresh variable to force main-figure refresh
        return refresh_main_fig + 1

    def recluster(self, filter_values=None, filter_column=None,
                  new_figerprints=None, new_chembl_ids=None, reload_data=False):
        if reload_data:
            return self.workflow.cluster()
        else:
            return self.workflow.re_cluster(filter_column, filter_values,
                                            new_figerprints=new_figerprints,
                                            new_chembl_ids=new_chembl_ids,
                                            n_clusters=self.n_clusters)

    def recluster_selection(self,
                           filter_value=None,
                           filter_column=None,
                           gradient_prop=None,
                           north_stars=None,
                           reload_data=False,
                           color_col='cluster'):

        if filter_value is not None:
            self.recluster(filter_values=filter_value,
                           filter_column=filter_column,
                           reload_data=reload_data)

        return self.create_graph(self.workflow.df_embedding,
                                 color_col=color_col,
                                 gradient_prop=gradient_prop,
                                 north_stars=north_stars)

    def create_graph(self, df, color_col='cluster', north_stars=None, gradient_prop=None):
        fig = go.Figure(layout={'colorscale': {}})

        if hasattr(df, 'compute'):
            ldf = df.compute()
        else:
            ldf = df

        moi_molregno = []
        if north_stars:
            moi_molregno = north_stars.split(",")

        moi_filter = ldf['id'].isin(moi_molregno)
        northstar_df = ldf[moi_filter]

        # Create a map with MoI and cluster to which they belong
        chemble_cluster_map = {}
        northstar_cluster = []
        if gradient_prop is not None:
            cmin = ldf[gradient_prop].min()
            cmax = ldf[gradient_prop].max()

            df_size = moi_filter
            # Compute size of northstar and normal points
            df_shape = df_size.copy()
            df_size = (df_size * 18) + DOT_SIZE
            df_shape = df_shape * 2

            x_data = ldf['x']
            y_data = ldf['y']
            cluster = ldf['cluster']
            customdata = ldf['id']
            grad_prop = ldf[gradient_prop]

            if self.enable_gpu:
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
            if self.enable_gpu:
                clusters = clusters.values_host

            if self.cluster_colors is None or len(self.cluster_colors) != len(clusters):
                self.cluster_colors = generate_colors(len(clusters))

            north_points = northstar_df['id']
            scatter_traces = []

            for cluster_id in clusters:
                query = 'cluster == ' + str(cluster_id)
                cdf = ldf.query(query)

                df_size = cdf['id'].isin(north_points)
                moi_present = False
                if df_size.shape[0] > 1:
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

                if self.enable_gpu:
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
        f_color = 'green' if self.enable_gpu else 'blue'

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
        return fig, northstar_cluster, json.dumps(chemble_cluster_map)

    def start(self, host=None, port=5000):
        return self.app.run_server(
            debug=False, use_reloader=False, host=host, port=port)

    def href_ify(self, molregno):
        return html.A(molregno,
                      href='https://www.ebi.ac.uk/chembl/compound_report_card/' + str(molregno),
                      target='_blank')

    def construct_molecule_detail(self, selected_points, display_properties,
                                  page, pageSize=10, chembl_ids=None):

        if not selected_points:
            return None, None

        # Create Table header
        table_headers = [html.Th("Molecular Structure", style={'width': '30%'}),
                         html.Th("Chembl"),
                         html.Th("smiles")]
        for prop in display_properties:
            table_headers.append(html.Th(prop))

        if chembl_ids:
            table_headers.append(html.Th('Cluster'))

        table_headers.append(html.Th(""))
        prop_recs = [html.Tr(table_headers)]

        if chembl_ids:
            selected_chembl_ids = chembl_ids
        else:
            selected_chembl_ids = []
            for point in selected_points['points'][((page - 1) * pageSize): page * pageSize]:
                if 'customdata' in point:
                    selected_chembl_ids.append(point['customdata'])

        props, selected_molecules = self.chem_data.fetch_props_by_molregno(
            selected_chembl_ids)
        all_props = []
        for k in props:
            all_props.append({"label": k, "value": k})

        for selected_molecule in selected_molecules:
            td = []
            selected_chembl_id = selected_molecule[1]
            smiles = selected_molecule[props.index('canonical_smiles')]

            mol = selected_molecule[props.index('molfile')]
            m = Chem.MolFromMolBlock(mol)

            drawer = Draw.rdMolDraw2D.MolDraw2DCairo(400, 200)
            drawer.SetFontSize(1.0)
            drawer.DrawMolecule(m)
            drawer.FinishDrawing()

            img_binary = "data:image/png;base64," + \
                base64.b64encode(drawer.GetDrawingText()).decode("utf-8")

            td.append(html.Img(src=img_binary))
            td.append(html.Td(self.href_ify(selected_chembl_id)))
            td.append(html.Td(smiles))
            for key in display_properties:
                td.append(html.Td(selected_molecule[props.index(key)]))

            if chembl_ids:
                td.append(html.Td(chembl_ids[selected_chembl_id]))
            td.append(html.Td(
                dbc.Button('Add as MoI',
                           id={'role': 'bt_star_candidate',
                               'chemblId': selected_chembl_id,
                               'molregno': str(selected_molecule[0])},
                           n_clicks=0)
            ))

            prop_recs.append(html.Tr(td))

        return html.Table(prop_recs, style={'width': '100%'}), all_props

    def constuct_layout(self):
        # TODO: avoid calling self.workflow.df_embedding
        fig, _, _ = self.create_graph(self.workflow.df_embedding)

        return html.Div([
            html.Div(className='row', children=[
                html.Div([dcc.Graph(id='main-figure', figure=fig), ],
                         className='nine columns',
                         style={'verticalAlign': 'text-top', }),
                html.Div([
                    html.Div(children=[
                        dcc.Markdown("""
                            **Molecule(s) of Interest (MoI)**

                            Please enter Chembl id."""), ]),
                    html.Div(className='row', children=[
                        dcc.Input(id='north_star', type='text', debounce=True),
                        dbc.Button('Highlight',
                                   id='bt_north_star', n_clicks=0,
                                   style={'marginLeft': 6, }),
                    ], style={'marginLeft': 0, 'marginTop': 18, }),

                    html.Div(children=[
                        dcc.Markdown("Set number of clusters")],
                        style={'marginTop': 18, 'marginLeft': 6},
                        className='row'),

                    html.Div(children=[
                        dcc.Input(id='sl_nclusters', value=self.n_clusters)],
                        style={'marginLeft': 6},
                        className='row'),

                    html.Div(children=[
                        dcc.Markdown("""
                            **Cluster Selection**

                            Click a point to select a cluster.
                        """)],
                        style={'marginTop': 18, 'marginLeft': 6},
                        className='row'),

                    html.Div(className='row', children=[
                        dcc.Input(id='selected_clusters', type='text'),
                        dbc.Button('Recluster',
                                   id='bt_recluster_clusters', n_clicks=0,
                                   style={'marginLeft': 6, }),
                    ], style={'marginLeft': 0, 'marginTop': 18, }),

                    html.Div(children=[
                        dcc.Markdown("""
                            **Selection Points**

                            Choose the lasso or rectangle tool in the graph's menu
                            bar and then select points in the graph.
                        """), ], style={'marginTop': 18, }),
                    dbc.Button('Recluster Selection',
                               id='bt_recluster_points', n_clicks=0),
                    html.Div(children=[
                             html.Div(id='selected_point_cnt'), ]),

                    html.Div(className='row', children=[
                        html.Div(children=[
                            dbc.Button("Close", id="bt_close"),
                            dbc.Modal([
                                dbc.ModalHeader("Close"),
                                dbc.ModalBody(
                                    dcc.Markdown("""
                                            Dashboard closed. Please return to the notebook.
                                        """),
                                ),
                                dbc.ModalFooter(dbc.Button(
                                    "Close", id="bt_close_dash", className="ml-auto")),
                            ], id="md_export"),
                        ]),

                        html.Div(children=[dbc.Button('Reload', id='bt_reset', n_clicks=0), ],
                                 style={'marginLeft': 18, }),
                    ], style={'marginLeft': 0, 'marginTop': 18, }),

                    html.Div(id='section_prop_gradient', children=[
                        html.Label([
                            "Select Molecular Property for color gradient",
                            dcc.Dropdown(id='sl_prop_gradient', multi=False,
                                         options=[{"label": p, "value": p} for p in IMP_PROPS],),
                        ], style={'marginTop': 18})],
                    ),

                ], className='three columns', style={'marginLeft': 18, 'marginTop': 90, 'verticalAlign': 'text-top', }),
            ]),
            html.Div(id='section_molecule_details', className='row', children=[
                html.Div(className='row', children=[
                    html.Div(id='section_display_properties', children=[
                        html.Label([
                            "Select Molecular Properties",
                            dcc.Dropdown(id='sl_mol_props', multi=True,
                                         options=[
                                             {'label': 'alogp', 'value': 'alogp'}],
                                         value=['alogp']),
                        ], style={'marginLeft': 60})],
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
                        style={
                            'paddingRight': 60, 'verticalAlign': 'text-bottom', 'text-align': 'right'}
                    ),
                ]),

                html.Div(className='row', children=[
                    html.Div(id='tb_selected_molecules', children=[],
                             style={'marginLeft': 60,
                                    'verticalAlign': 'text-top'}
                             ),
                ])
            ], style={'display': 'none'}),

            html.Div(id='refresh_main_fig', style={'display': 'none'}),
            html.Div(id='northstar_cluster', style={'display': 'none'}),
            html.Div(id='hidden_northstar', style={'display': 'none'}),
            html.Div(id='north_star_clusterid_map', style={'display': 'none'}),
            html.Div(id='recluster_error'),
            html.Div(id='mol_selection_error')
        ])

    @report_ui_error(5)
    def handle_molecule_selection(self, mf_selected_data, selected_columns,
                                  sl_prop_gradient, prev_click, next_click,
                                  north_star_clusterid_map,
                                  current_page):
        comp_id, event_type = self._fetch_event_data()

        if (not mf_selected_data) and comp_id != 'north_star_clusterid_map':
            raise dash.exceptions.PreventUpdate

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
        elif comp_id == 'north_star_clusterid_map' and event_type == 'children':
            chembl_ids = json.loads(north_star_clusterid_map)
            if len(chembl_ids) == 0:
                raise dash.exceptions.PreventUpdate

        if selected_columns and sl_prop_gradient:
            if sl_prop_gradient not in selected_columns:
                selected_columns.append(sl_prop_gradient)
        module_details, all_props = self.construct_molecule_detail(
            mf_selected_data, selected_columns, current_page,
            pageSize=PAGE_SIZE, chembl_ids=chembl_ids)

        if chembl_ids:
            last_page = ''
        else:
            last_page = ' of ' + str(len(mf_selected_data['points'])//PAGE_SIZE)

        return module_details, all_props, current_page, last_page, {'display': 'block'}, dash.no_update

    def handle_data_selection(self, mf_click_data, mf_selected_data,
                              bt_cluster_clicks, bt_point_clicks,
                              northstar_cluster,
                              curr_clusters):
        comp_id, event_type = self._fetch_event_data()
        selected_clusters = ''
        selected_point_cnt = ''

        if comp_id == 'main-figure' and event_type == 'clickData':
            # Event - On selecting cluster on the main scatter plot
            if not curr_clusters:
                clusters = []
            else:
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

    def handle_mark_north_star(self, bt_north_star_click, north_star, hidden_northstar):
        comp_id, event_type = self._fetch_event_data()

        if event_type != 'n_clicks' or dash.callback_context.triggered[0]['value'] == 0:
            raise dash.exceptions.PreventUpdate

        selected_north_star = []
        selected_north_star_mol_reg_id = []

        if north_star:
            selected_north_star = north_star.split(",")
            if hidden_northstar:
                selected_north_star_mol_reg_id = hidden_northstar.split(",")
            else:
                selected_north_star_mol_reg_id = [
                    str(row[0]) for row in self.chem_data.fetch_molregno_by_chemblId(selected_north_star)]

        comp_detail = json.loads(comp_id)
        selected_chembl_id = comp_detail['chemblId']

        if selected_chembl_id not in selected_north_star:
            selected_north_star.append(selected_chembl_id)
            selected_north_star_mol_reg_id.append(comp_detail['molregno'])
        return ','.join(selected_north_star), ','.join(selected_north_star_mol_reg_id)

    @report_ui_error(3)
    def handle_re_cluster(self, bt_cluster_clicks, bt_point_clicks, bt_north_star_clicks,
                          north_star_hidden, sl_prop_gradient, sl_nclusters,  refresh_main_fig,
                          selected_clusters, selected_points, north_star):

        comp_id, event_type = self._fetch_event_data()
        self.n_clusters = int(sl_nclusters)
        filter_values = None
        filter_column = None
        reload_data = False

        if selected_clusters and comp_id == 'bt_recluster_clusters' and event_type == 'n_clicks':
            filter_values = list(map(int, selected_clusters.split(",")))
            filter_column = 'cluster'

        elif selected_points and comp_id == 'bt_recluster_points' and event_type == 'n_clicks':
            filter_values = []
            for point in selected_points['points']:
                if 'customdata' in point:
                    filter_values.append(point['customdata'])
            filter_column = 'id'

        elif comp_id == 'bt_north_star' and event_type == 'n_clicks':
            north_star_hidden = self.update_new_chembl(north_star)
            if not north_star_hidden:
                raise dash.exceptions.PreventUpdate

        elif comp_id == 'hidden_northstar' and event_type == 'value':
            if not north_star_hidden:
                raise dash.exceptions.PreventUpdate

        elif comp_id == 'refresh_main_fig' and event_type == 'value':
            reload_data = True

        # else:
        #     raise dash.exceptions.PreventUpdate

        figure, northstar_cluster, chembl_clusterid_map = self.recluster_selection(
            filter_value=filter_values,
            filter_column=filter_column,
            gradient_prop=sl_prop_gradient,
            north_stars=north_star_hidden,
            color_col='cluster',
            reload_data = reload_data)

        if north_star_hidden is None:
            northstar_cluster = ''
        else:
            northstar_cluster = north_star_hidden

        return figure, ','.join(northstar_cluster), chembl_clusterid_map, ''

    def update_new_chembl(self, north_stars, radius=2, nBits=512):
        north_stars = list(map(str.strip, north_stars.split(',')))
        north_stars = list(map(str.upper, north_stars))
        molregnos = [row[0] for row in self.chem_data.fetch_molregno_by_chemblId(north_stars)]

        # TODO: Avoid using self.workflow.df_embedding
        df = self.workflow.df_embedding
        df['id_exists'] = df['id'].isin(molregnos)

        ldf = df.query('id_exists == True')
        ldf = ldf.compute()
        df.drop(['id_exists'], axis=1)

        missing_molregno = set(molregnos).difference(ldf['id'].to_array())
        # CHEMBL10307, CHEMBL103071, CHEMBL103072
        if missing_molregno:
            missing_molregno = list(missing_molregno)
            ldf = self.chem_data.fetch_props_df_by_molregno(missing_molregno)

            if ldf.shape[0] > 0:

                smiles = []
                for i in range(0, ldf.shape[0]):
                    smiles.append(
                        ldf.iloc[i]['canonical_smiles'].to_array()[0])

                morgan_fingerprint = MorganFingerprint()
                results = list(morgan_fingerprint.transform_many(smiles))
                fingerprints = cupy.stack(results).astype(np.float32)
                tdf = self.recluster(df, fingerprints, missing_molregno)
                if tdf:
                    df = tdf
                else:
                    return None

        return " ,".join(list(map(str, molregnos)))
