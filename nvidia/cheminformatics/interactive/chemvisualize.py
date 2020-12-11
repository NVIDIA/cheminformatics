# Copyright 2020 NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import base64
import logging
import pandas
from rdkit import Chem
from rdkit.Chem import Draw
import numpy as np

import cudf
import cuml
import cupy

import sklearn.cluster

# from jupyter_dash import JupyterDash
import plotly.graph_objects as go
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State, ALL

from nvidia.cheminformatics.chembldata import ChEmblData
from nvidia.cheminformatics.fingerprint import MorganFingerprint


logger = logging.getLogger(__name__)

external_stylesheets = [
    'https://codepen.io/chriddyp/pen/bWLwgP.css', dbc.themes.BOOTSTRAP]

main_fig_height = 700
CHEMBL_DB = '/data/db/chembl_27.db'
PAGE_SIZE = 10
DOT_SIZE = 5
IMP_PROPS = [
    'alogp',
    'aromatic_rings',
    'full_mwt',
    'psa',
    'rtb']

COLORS = ["#406278", "#e32636", "#9966cc", "#cd9575", "#915c83", "#008000",
          "#ff9966", "#848482", "#8a2be2", "#de5d83", "#800020", "#e97451",
          "#5f9ea0", "#36454f", "#008b8b", "#e9692c", "#f0b98d", "#ef9708",
          "#0fcfc0", "#9cded6", "#d5eae7", "#f3e1eb", "#f6c4e1", "#f79cd4"]


class ChemVisualization:

    def __init__(self, df, workflow, gpu=True):
        self.enable_gpu = gpu
        self.app = dash.Dash(
            __name__, external_stylesheets=external_stylesheets)
        self.df = df
        self.df['id'] = self.df.index

        self.workflow = workflow
        self.n_clusters = workflow.n_clusters

        self.chem_data = ChEmblData()

        self.molregno = self.df.index
        self.orig_df = df.copy()

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
             Output('north_star_clusterid_map', 'children')],
            [Input('bt_recluster_clusters', 'n_clicks'),
             Input('bt_recluster_points', 'n_clicks'),
             Input('bt_north_star', 'n_clicks'),
             Input('hidden_northstar', 'value'),
             Input('sl_prop_gradient', 'value'),
             Input('sl_nclusters', 'value'), ],
            [State("selected_clusters", "value"),
             State("main-figure", "selectedData"),
             State('north_star', 'value'), ])(self.handle_re_cluster)

        # Register callbacks for selection inside main figure to update module details
        self.app.callback(
            [Output('tb_selected_molecules', 'children'),
             Output('sl_mol_props', 'options'),
             Output('current_page', 'children'),
             Output('total_page', 'children'),
             Output('section_molecule_details', 'style')],
            [Input('main-figure', 'selectedData'),
             Input('sl_mol_props', 'value'),
             Input('sl_prop_gradient', 'value'),
             Input('bt_page_prev', 'n_clicks'),
             Input('bt_page_next', 'n_clicks'),
             Input('north_star_clusterid_map', 'children')],
            State('current_page', 'children'))(self.handle_molecule_selection)

        self.app.callback(
            Output("hidden1", "children"),
            [Input("bt_reset", "n_clicks")])(self.handle_reset)

        self.app.callback(
            [Output('north_star', 'value'),
             Output('hidden_northstar', 'value')],
            [Input({'role': 'bt_star_candidate', 'chemblId': ALL, 'molregno': ALL}, 'n_clicks')],
            [State('north_star', 'value'),
             State('hidden_northstar', 'value')])(self.handle_mark_north_star)

    def re_cluster(self, gdf, new_figerprints=None, new_chembl_ids=None):
        return self.workflow.re_cluster(gdf,
                                        new_figerprints=None,
                                        new_chembl_ids=None,
                                        n_clusters=self.n_clusters)

    def recluster_nofilter(self, df, gradient_prop, north_stars=None):
        tdf = self.re_cluster(df)
        if tdf is not None:
            self.df = tdf
        return self.create_graph(self.df, color_col='cluster',
                                 gradient_prop=gradient_prop, north_stars=north_stars)

    def recluster_selected_clusters(self, df, values, gradient_prop, north_stars=None):
        df_clusters = df['cluster'].isin(values)
        # filters = df_clusters.values

        df['filter_col'] = df_clusters
        tdf = df.query('filter_col == True')

        tdf = self.re_cluster(tdf)

        if tdf is not None:
            self.df = tdf
        return self.create_graph(tdf, color_col='cluster',
                                 gradient_prop=gradient_prop,
                                 north_stars=north_stars)

    def recluster_selected_points(self, df, values, gradient_prop, north_stars=None):
        df_clusters = df['id'].isin(values)

        # filters = df_clusters.values
        df['filter_col'] = df_clusters
        tdf = df.query('filter_col == True')

        tdf = self.re_cluster(tdf)
        if tdf is not None:
            self.df = tdf
        return self.create_graph(self.df, color_col='cluster',
                                 gradient_prop=gradient_prop,
                                 north_stars=north_stars)

    def create_graph(self, df, color_col='cluster', north_stars=None, gradient_prop=None):
        fig = go.Figure(layout={'colorscale': {}})

        ldf = df.compute()

        moi_molregno = []
        if north_stars:
            moi_molregno = north_stars.split(",")

        moi_filter = ldf.index.isin(moi_molregno)
        northstar_df = ldf[moi_filter]

        # Create a map with MoI and cluster to which they belong
        chemble_cluster_map = {}
        # if north_stars:
        #     print('====>>>>', north_stars)
        #     chemble_cluster_map = dict(zip(northstar_df['chembl_id'].to_array(),
        #                                    northstar_df['cluster'].to_array().tolist()))

        northstar_cluster = []
        if gradient_prop is not None:
            cmin = ldf[gradient_prop].min()
            cmax = ldf[gradient_prop].max()

            df_size = moi_filter
            # Compute size of northstar and normal points
            df_shape = df_size.copy()
            df_size = (df_size * 18) + DOT_SIZE
            df_shape = df_shape * 2

            fig.add_trace(go.Scattergl({
                'x': ldf['x'].to_array(),
                'y': ldf['y'].to_array(),
                'text': ldf['cluster'].to_array(),
                'customdata': ldf.index.to_array(),
                'mode': 'markers',
                'showlegend': False,
                'marker': {
                    'size': df_size.to_array(),
                    'symbol': df_shape.to_array(),
                    'color': ldf[gradient_prop].to_array(),
                    'colorscale': 'Viridis',
                    'showscale': True,
                    'cmin': cmin,
                    'cmax': cmax,
                }
            }))
        else:
            if self.enable_gpu:
                colors = ldf[color_col].unique().values_host
            else:
                colors = ldf[color_col].unique()

            north_points = northstar_df.index
            scatter_traces = []
            for cluster_id in colors:
                query = 'cluster == ' + str(cluster_id)
                cdf = ldf.query(query)

                moi_present = False

                df_size = cdf['id'].isin(north_points)
                if df_size.unique().shape[0] > 1:
                    northstar_cluster.append(str(cluster_id))
                    moi_present = True

                # Compute size of northstar and normal points
                df_shape = df_size.copy()
                df_size = (df_size * 18) + DOT_SIZE
                df_shape = df_shape * 2
                if self.enable_gpu:
                    scatter_trace = go.Scattergl({
                        'x': cdf['x'].to_array(),
                        'y': cdf['y'].to_array(),
                        'text': cdf['cluster'].to_array(),
                        'customdata': cdf.index.to_array(),
                        'name': 'Cluster ' + str(cluster_id),
                        'mode': 'markers',
                        'marker': {
                            'size': df_size.to_array(),
                            'symbol': df_shape.to_array(),
                            'color': COLORS[cluster_id % len(COLORS)],
                        },
                    })
                else:
                    scatter_trace = go.Scattergl({
                        'x': cdf['x'],
                        'y': cdf['y'],
                        'text': cdf['cluster'],
                        'customdata': cdf.index,
                        'name': 'Cluster ' + str(cluster_id),
                        'mode': 'markers',
                        'marker': {
                            'size': df_size,
                            'symbol': df_shape,
                            'color': COLORS[cluster_id % len(COLORS)],
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
        #TODO: Get molregno
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
            for point in selected_points['points'][((page-1)*pageSize + 1): page * pageSize]:
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
        fig, _, _ = self.create_graph(self.df)

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

                        html.Div(children=[html.A(dbc.Button('Reload', id='bt_reset'), href='/'), ],
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

            html.Div(id='hidden1', style={'display': 'none'}),
            html.Div(id='northstar_cluster', style={'display': 'none'}),
            html.Div(id='hidden_northstar', style={'display': 'none'}),
            html.Div(id='north_star_clusterid_map', style={'display': 'none'}),
        ])

    def handle_reset(self, recluster_nofilter):
        self.df = self.orig_df.copy()

        self.molregno = self.df.index
        self.df['id'] = self.df.index
        self.re_cluster(self.df)

    def handle_molecule_selection(self, mf_selected_data, selected_columns,
                                  sl_prop_gradient, prev_click, next_click,
                                  north_star_clusterid_map,
                                  current_page):
        if not dash.callback_context.triggered:
            raise dash.exceptions.PreventUpdate
        comp_id, event_type = \
            dash.callback_context.triggered[0]['prop_id'].split('.')

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
            last_page = ' of ' + \
                str(len(mf_selected_data['points'])//PAGE_SIZE)
        return module_details, all_props, current_page, last_page, {'display': 'block'}

    def handle_data_selection(self, mf_click_data, mf_selected_data,
                              bt_cluster_clicks, bt_point_clicks,
                              northstar_cluster,
                              curr_clusters):
        if not dash.callback_context.triggered:
            raise dash.exceptions.PreventUpdate
        comp_id, event_type = \
            dash.callback_context.triggered[0]['prop_id'].split('.')
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
            # selected_clusters = ','.join(map(str, clusters))
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
        if not dash.callback_context.triggered:
            raise dash.exceptions.PreventUpdate

        comp_id, event_type = \
            dash.callback_context.triggered[0]['prop_id'].split('.')

        if event_type != 'n_clicks' or dash.callback_context.triggered[0]['value'] == 0:
            raise dash.exceptions.PreventUpdate

        if not north_star:
            selected_north_star = []
            selected_north_star_mol_reg_id = []
        else:
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

    def handle_re_cluster(self, bt_cluster_clicks, bt_point_clicks, bt_north_star_clicks,
                          north_star_hidden, sl_prop_gradient, sl_nclusters,
                          curr_clusters, mf_selected_data, north_star):
        if not dash.callback_context.triggered:
            raise dash.exceptions.PreventUpdate

        comp_id, event_type = \
            dash.callback_context.triggered[0]['prop_id'].split('.')

        self.n_clusters = int(sl_nclusters)

        if comp_id == 'bt_recluster_clusters' and event_type == 'n_clicks':
            if not curr_clusters:
                figure, northstar_cluster, chembl_clusterid_map = \
                    self.recluster_nofilter(self.df,
                                            sl_prop_gradient,
                                            north_stars=north_star_hidden)
            else:
                clusters = list(map(int, curr_clusters.split(",")))
                figure, northstar_cluster, chembl_clusterid_map = \
                    self.recluster_selected_clusters(self.df,
                                                     clusters,
                                                     sl_prop_gradient,
                                                     north_stars=north_star_hidden)

        elif comp_id == 'bt_recluster_points' and event_type == 'n_clicks':
            if not mf_selected_data:
                figure, northstar_cluster, chembl_clusterid_map = \
                    self.recluster_nofilter(
                        self.df, sl_prop_gradient, north_stars=north_star_hidden)
            else:
                points = []
                for point in mf_selected_data['points']:
                    points.append(point['customdata'])
                figure, northstar_cluster, chembl_clusterid_map = \
                    self.recluster_selected_points(self.df, points,
                                                   sl_prop_gradient,
                                                   north_stars=north_star_hidden)

        elif (comp_id == 'sl_prop_gradient' and event_type == 'value'):
            figure, _, chembl_clusterid_map = self.create_graph(
                self.df, gradient_prop=sl_prop_gradient, north_stars=north_star_hidden)
            northstar_cluster = curr_clusters.split(',')
        elif comp_id == 'bt_north_star' and event_type == 'n_clicks':
            molregnos = self.update_new_chembl(north_star)
            if molregnos:
                figure, northstar_cluster, chembl_clusterid_map = self.create_graph(
                    self.df, gradient_prop=sl_prop_gradient, north_stars=molregnos)
            else:
                raise dash.exceptions.PreventUpdate
        elif comp_id == 'hidden_northstar' and event_type == 'value':
            if north_star_hidden:
                figure, northstar_cluster, chembl_clusterid_map = self.create_graph(
                    self.df, gradient_prop=sl_prop_gradient, north_stars=north_star_hidden)
            else:
                raise dash.exceptions.PreventUpdate
        else:
            raise dash.exceptions.PreventUpdate

        return figure, ','.join(northstar_cluster), chembl_clusterid_map

    def update_new_chembl(self, north_stars, radius=2, nBits=512):
        north_stars = list(map(str.strip, north_stars.split(',')))
        north_stars = list(map(str.upper, north_stars))
        molregnos = [row[0] for row in self.chem_data.fetch_molregno_by_chemblId(north_stars)]

        self.df['id_exists'] = self.df.index.isin(molregnos)

        ldf = self.df.query('id_exists == True')
        ldf = ldf.compute()
        self.df.drop(['id_exists'], axis=1)

        missing_molregno = set(molregnos).difference(ldf.index.to_array())
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
                tdf = self.re_cluster(self.df, fingerprints, missing_molregno)
                if tdf:
                    self.df = tdf
                else:
                    return None

        return " ,".join(list(map(str, molregnos)))
