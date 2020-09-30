# Copyright 2020 NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0

import io
import time
import json
import base64
import pandas as pd
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from chembl_webresource_client.new_client import new_client
from chembl_webresource_client.utils import utils
from PIL import Image
import numpy as np
import os, sys, wget, gzip
import hashlib
from ftplib import FTP
import matplotlib.pyplot as plt
import sqlite3
from contextlib import closing

import cudf, cuml
import cupy
from cuml import KMeans, UMAP

# from jupyter_dash import JupyterDash
from flask import request
import plotly.graph_objects as go
import dash
import dask.dataframe as dd
import dask.bag as db
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import dash_table as table

from dask import delayed
from dash.dependencies import Input, Output, State, ALL, MATCH

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css', dbc.themes.BOOTSTRAP]

main_fig_height = 700
CHEMBL_DB='/data/db/chembl_27.db'
PAGE_SIZE = 10
IMP_PROPS = [
    'alogp',
    'aromatic_rings',
    'full_mwt',
    'psa',
    'rtb']

class ChemVisualization:

    def __init__(self, df, n_clusters, chembl_ids):
        self.app = dash.Dash(
            __name__, external_stylesheets=external_stylesheets)
        self.df = df
        self.n_clusters = n_clusters
        self.chembl_ids = chembl_ids

        # Fetch relavant properties from database.
        self.prop_df = self.create_dataframe_molecule_properties(chembl_ids)

        self.df['chembl_id'] = chembl_ids
        self.df['id'] = self.df.index
        self.orig_df = df.copy()

        # initialize UMAP
        self.umap = UMAP(n_neighbors=100,
                a=1.0,
                b=1.0,
                learning_rate=1.0)

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
            [State("selected_clusters", "value")]) (self.handle_data_selection)

        # Register callbacks for buttons for reclustering selected data
        self.app.callback(
            [Output('main-figure', 'figure'),
             Output('northstar_cluster', 'children')],
            [Input('bt_recluster_clusters', 'n_clicks'),
             Input('bt_recluster_points', 'n_clicks'),
             Input('bt_north_star', 'n_clicks'),
             Input('north_star', 'value'),
             Input('sl_prop_gradient', 'value'),
             Input('sl_nclusters', 'value')],
            [State("selected_clusters", "value"),
             State("main-figure", "selectedData")]) (self.handle_re_cluster)

        # Register callbacks for selection inside main figure to update module details
        self.app.callback(
            [Output('tb_selected_molecules', 'children'),
             Output('sl_mol_props', 'options'),
             Output("current_page", "children"),
             Output("total_page", "children"),
             Output('section_molecule_details', 'style')],
            [Input('main-figure', 'selectedData'),
             Input('sl_mol_props', 'value'),
             Input('sl_prop_gradient', 'value'),
             Input("bt_page_prev", "n_clicks"),
             Input("bt_page_next", "n_clicks")],
             State("current_page", "children")) (self.handle_molecule_selection)

        self.app.callback(
            Output("hidden1", "children"),
            [Input("bt_reset", "n_clicks")]) (self.handle_reset)

        self.app.callback(
            Output('north_star', 'value'),
            [Input({'role': 'bt_star_candidate', 'index': ALL}, 'n_clicks')],
            State('north_star', 'value')) \
                (self.handle_mark_north_star)

    def MorganFromSmiles(self, smiles, radius=2, nBits=512):
        m = Chem.MolFromSmiles(smiles)
        fp = AllChem.GetMorganFingerprintAsBitVect(m, radius=radius, nBits=nBits)
        ar = cupy.array(fp)
        return ar

    def re_cluster(self, gdf, new_figerprints=None, new_chembl_ids=None):
        if gdf.shape[0] == 0:
            return None

        # Before reclustering remove all columns that may interfere  
        ids = gdf['id'] 
        chembl_ids = gdf['chembl_id']

        gdf.drop(['x', 'y', 'cluster', 'id', 'chembl_id'], inplace=True)
        if new_figerprints is not None and new_chembl_ids is not None:
            # Add new figerprints and chEmblIds before reclustering
            fp_df = cudf.DataFrame(new_figerprints, columns=gdf.columns)
            gdf = gdf.append(fp_df, ignore_index=True)
            chembl_ids = chembl_ids.append(
                cudf.Series(new_chembl_ids), ignore_index=True)

        kmeans_float = KMeans(n_clusters=self.n_clusters)
        kmeans_float.fit(gdf)

        Xt = self.umap.fit_transform(gdf)

        # Add back the column required for plotting and to correlating data 
        # between re-clustering 
        gdf.add_column('x', Xt[0].to_array())
        gdf.add_column('y', Xt[1].to_array())
        gdf.add_column('id', gdf.index)
        gdf.add_column('chembl_id', chembl_ids)
        gdf.add_column('cluster', kmeans_float.labels_.to_array())
        return gdf

    def recluster_nofilter(self, df, gradient_prop, north_stars=None):
        tdf = self.re_cluster(df)
        if tdf is not None:
            self.df = tdf
        return self.create_graph(self.df, color_col='cluster', 
            gradient_prop=gradient_prop, north_stars=north_stars)

    def recluster_selected_clusters(self, df, values, gradient_prop, north_stars=None):
        df_clusters = df['cluster'].isin(values)
        filters = df_clusters.values

        tdf = df[filters.get()]
        tdf = self.re_cluster(tdf)
        if tdf is not None:
            self.df = tdf
        return self.create_graph(self.df, color_col='cluster', 
            gradient_prop=gradient_prop, north_stars=north_stars)

    def recluster_selected_points(self, df, values, gradient_prop, north_stars=None):
        df_clusters = df['id'].isin(values)
        filters = df_clusters.values

        tdf = df[filters.get()]
        tdf = self.re_cluster(tdf)
        if tdf is not None:
            self.df = tdf
        return self.create_graph(self.df, color_col='cluster', 
            gradient_prop=gradient_prop, north_stars=north_stars)

    def create_graph(self, df, color_col='cluster', north_stars=None, gradient_prop=None):
        fig = go.Figure(layout = {'colorscale' : {}})
        ldf = df.merge(self.prop_df, on='chembl_id')

        cmin = cmax = None
        if gradient_prop is not None:
            cmin = ldf[gradient_prop].min()
            cmax = ldf[gradient_prop].max()

        north_points = []
        if north_stars:
            for chemblid in north_stars.split(","):
                chemblid = chemblid.strip()
                if chemblid in self.chembl_ids:
                    north_points.append(self.chembl_ids.index(chemblid))

        northstar_cluster = []
        for cluster_id in ldf[color_col].unique().values_host:
            query = 'cluster == ' + str(cluster_id)
            cdf = ldf.query(query)

            df_size = cdf['id'].isin(north_points)
            if df_size.unique().shape[0] > 1:
                northstar_cluster.append(str(cluster_id))

            # Compute size of northstar and normal points
            df_shape = df_size.copy()
            df_size = (df_size * 18) + 6
            df_shape = df_shape * 2
            if gradient_prop is not None:

                fig.add_trace(
                    go.Scattergl({
                        'x': cdf['x'].to_array(),
                        'y': cdf['y'].to_array(),
                        'text': cdf['chembl_id'].to_array(),
                        'customdata': cdf['id'].to_array(),
                        'name': 'Cluster ' + str(cluster_id),
                        'mode': 'markers',
                        'showlegend': False,
                        'marker': {
                            'size': df_size.to_array(),
                            'symbol': df_shape.to_array(),
                            'color': cdf[gradient_prop].to_array(),
                            'colorscale': 'Viridis',
                            'showscale': True,
                            'cmin': cmin,
                            'cmax': cmax,
                        }
                }))
            else:
                fig.add_trace(
                    go.Scattergl({
                        'x': cdf['x'].to_array(),
                        'y': cdf['y'].to_array(),
                        'text': cdf['chembl_id'].to_array(),
                        'customdata': cdf['id'].to_array(),
                        'name': 'Cluster ' + str(cluster_id),
                        'mode': 'markers',
                        'marker': {
                            'size': df_size.to_array(),
                            'symbol': df_shape.to_array(),
                        }
                }))

        fig.update_layout(
            showlegend=True, clickmode='event', height=main_fig_height, 
                title='Clusters', dragmode='select',
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


    def href_ify(self, chemblid):
        return html.A(chemblid, href='https://www.ebi.ac.uk/chembl/compound_report_card/' + chemblid,
                      target='_blank')

    #TODO: remove self.selected_chembl_id
    def construct_molecule_detail(self, selected_points, display_properties, page, pageSize=10):
        # Create Table header
        table_headers = [html.Th("Molecular Structure", style={'width': '30%'}),
              html.Th("Chembl"),
              html.Th("smiles")]
        for prop in display_properties:
            table_headers.append(html.Th(prop))

        table_headers.append(html.Th(""))
        prop_recs = [html.Tr(table_headers)]

        selected_chembl_ids = []
        for point in selected_points['points'][((page-1)*pageSize + 1): page * pageSize]:
            selected_chembl_ids.append(point['text'])

        props, selected_molecules = self.fetch_molecule_properties(selected_chembl_ids)
        all_props = []
        for k in props:
            all_props.append({"label": k, "value": k})

        for selected_molecule in selected_molecules:
            td = []
            selected_chembl_id = selected_molecule[0]
            smiles = selected_molecule[props.index('canonical_smiles')]

            mol = selected_molecule[props.index('molfile')]
            m = Chem.MolFromMolBlock(mol)

            drawer = Draw.rdMolDraw2D.MolDraw2DCairo(400, 200)
            drawer.SetFontSize(1.0)
            drawer.DrawMolecule(m)
            drawer.FinishDrawing()

            img_binary="data:image/png;base64," + \
                base64.b64encode(drawer.GetDrawingText()).decode("utf-8")

            td.append(html.Img(src=img_binary))
            td.append(html.Td(self.href_ify(selected_chembl_id)))
            td.append(html.Td(smiles))
            for key in display_properties:
                td.append(html.Td(selected_molecule[props.index(key)]))
            td.append(html.Td(
                dbc.Button('Add to MoI', \
                    id={'role': 'bt_star_candidate', 'index': selected_chembl_id}, n_clicks=0)
            ))
            prop_recs.append(html.Tr(td))
        return  html.Table(prop_recs, style={'width': '100%'}), all_props

    def constuct_layout(self):
        fig, northstart_cluster = self.create_graph(self.df)

        return html.Div([
            html.Div(className='row', children=[
                html.Div([dcc.Graph(id='main-figure', figure=fig),], 
                    className='nine columns', 
                    style={'verticalAlign': 'text-top',}),
                html.Div([                    
                    html.Div(children=[
                        dcc.Markdown("""
                            **Molecule(s) of Interest (MoI)**

                            Please enter Chembl id."""), ]),
                    html.Div(className='row', children=[
                        dcc.Input(id='north_star', type='text', debounce=True),
                        dbc.Button('Highlight', 
                            id='bt_north_star', n_clicks=0,
                            style={'marginLeft': 6,}),
                        ], style={'marginLeft': 0, 'marginTop': 18,}),

                    html.Div(id='section_nclusters', children=[
                        html.Label([
                            "Set number of clusters",
                            dcc.Dropdown(id='sl_nclusters', 
                                         multi=False,
                                         options=[{"label": p, "value": p} for p in range(2,10)],
                                         value=self.n_clusters
                                        ),
                        ], style={'marginTop': 6})], 
                    ),

                    html.Div(children=[
                        dcc.Markdown("""
                            **Cluster Selection**

                            Click a point to select a cluster.
                        """)], style={'marginTop': 18,}),

                    html.Div(className='row', children=[
                        dcc.Input(id='selected_clusters', type='text'),
                        dbc.Button('Recluster', 
                            id='bt_recluster_clusters', n_clicks=0,
                            style={'marginLeft': 6,}),
                        ], style={'marginLeft': 0, 'marginTop': 18,}),

                    html.Div(children=[
                        dcc.Markdown("""
                            **Selection Points**

                            Choose the lasso or rectangle tool in the graph's menu
                            bar and then select points in the graph.
                        """),], style={'marginTop': 18,}),
                    dbc.Button('Recluster Selection', 
                        id='bt_recluster_points', n_clicks=0),
                    html.Div(children=[
                             html.Div(id='selected_point_cnt'),]),

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
                                    dbc.ModalFooter(dbc.Button("Close", id="bt_close_dash", className="ml-auto")),
                                ], id="md_export"),
                        ]),

                        html.Div(children=[html.A(dbc.Button('Reload', id='bt_reset'), href='/'),], 
                                style={'marginLeft': 18,}),
                    ], style={'marginLeft': 0, 'marginTop': 18,}),

                    html.Div(id='section_prop_gradient', children=[
                        html.Label([
                            "Select Molecular Property for color gradient",
                            dcc.Dropdown(id='sl_prop_gradient', multi=False,
                                options=[{"label": p, "value": p} for p in IMP_PROPS],),
                        ], style={'marginTop': 18})], 
                    ),

                ], className='three columns', style={'marginLeft': 18, 'marginTop': 90, 'verticalAlign': 'text-top',}),
            ]),
            html.Div(id='section_molecule_details', className='row', children=[
                html.Div(className='row', children=[
                    html.Div(id='section_display_properties', children=[
                        html.Label([
                            "Select Molecular Properties",
                            dcc.Dropdown(id='sl_mol_props', multi=True,
                                options=[{'label': 'alogp', 'value': 'alogp'}],
                                value=['alogp']),
                        ], style={'marginLeft': 60})],
                        className='nine columns', 
                    ),
                    html.Div(children=[
                            dbc.Button("<", id="bt_page_prev", style={"height": "25px"}),
                            html.Span(children=1, id='current_page', style={"paddingLeft": "6px"}),
                            html.Span(children=' of 1', id='total_page', style={"paddingRight": "6px"}),
                            dbc.Button(">", id="bt_page_next", style={"height": "25px"})
                        ],
                        className='three columns',
                        style={'paddingRight': 60, 'verticalAlign': 'text-bottom', 'text-align': 'right'}
                    ),
                ]),

                html.Div(className='row', children=[
                    html.Div(id='tb_selected_molecules', children=[],
                        style={'marginLeft': 60, 'verticalAlign': 'text-top'}
                    ), 
                ])
            ], style={'display':'none'}),

            html.Div(id='hidden1', style={'display':'none'}),
            html.Div(id='northstar_cluster', style={'display':'none'})
        ])

    def handle_reset(self, recluster_nofilter):
        self.df = self.orig_df.copy()

    def handle_molecule_selection(self, mf_selected_data, selected_columns,
            sl_prop_gradient, prev_click, next_click, current_page):
        if not dash.callback_context.triggered or not mf_selected_data:
            raise dash.exceptions.PreventUpdate
        comp_id, event_type = \
            dash.callback_context.triggered[0]['prop_id'].split('.')

        module_details = None
        # Code to support pagination
        if comp_id == 'bt_page_prev' and event_type == 'n_clicks':
            if current_page == 1:
                raise dash.exceptions.PreventUpdate
            current_page -= 1
        elif comp_id == 'bt_page_next' and event_type == 'n_clicks':
            if len(mf_selected_data['points']) < PAGE_SIZE * (current_page + 1):
                raise dash.exceptions.PreventUpdate
            current_page += 1

        if selected_columns and sl_prop_gradient:
            if sl_prop_gradient not in selected_columns:
                selected_columns.append(sl_prop_gradient)

        module_details, all_props = self.construct_molecule_detail(
            mf_selected_data, selected_columns, current_page, pageSize=PAGE_SIZE)
        
        last_page = ' of ' + str(len(mf_selected_data['points'])//PAGE_SIZE)
        return module_details, all_props, current_page, last_page, {'display':'block'}

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
                cluster = point['curveNumber']
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
            clusters = {point['curveNumber'] for point in points}
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

    def handle_mark_north_star(self, bt_north_star_click, north_star):
        if not dash.callback_context.triggered:
            raise dash.exceptions.PreventUpdate

        comp_id, event_type = \
            dash.callback_context.triggered[0]['prop_id'].split('.')

        if event_type != 'n_clicks' or dash.callback_context.triggered[0]['value'] == 0:
            raise dash.exceptions.PreventUpdate

        if not north_star:
            selected_north_star = []
        else:
            selected_north_star = north_star.split(",") 

        comp_detail = json.loads(comp_id)
        selected_chembl_id = comp_detail['index']

        if selected_chembl_id not in selected_north_star and \
            selected_chembl_id in self.chembl_ids:
            selected_north_star.append(selected_chembl_id)

        return ','.join(selected_north_star)

    def handle_re_cluster(self, bt_cluster_clicks, bt_point_clicks, bt_north_star_clicks,
                          north_star, sl_prop_gradient, sl_nclusters, curr_clusters, mf_selected_data):
        if not dash.callback_context.triggered:
            raise dash.exceptions.PreventUpdate

        comp_id, event_type = \
            dash.callback_context.triggered[0]['prop_id'].split('.')

        self.n_clusters = sl_nclusters

        if comp_id == 'bt_recluster_clusters' and event_type == 'n_clicks':
            if not curr_clusters:
                figure, northstar_cluster = self.recluster_nofilter(
                    self.df, sl_prop_gradient, north_stars=north_star)
            else:
                clusters = list(map(int, curr_clusters.split(",")))
                figure, northstar_cluster = self.recluster_selected_clusters(
                    self.df, clusters, sl_prop_gradient, north_stars=north_star)
            
        elif comp_id == 'bt_recluster_points' and event_type == 'n_clicks':
            if not mf_selected_data:
                figure, northstar_cluster = self.recluster_nofilter(
                    self.df, sl_prop_gradient, north_stars=north_star)
            else:
                points = []
                for point in mf_selected_data['points']:
                    points.append(point['customdata'])
                figure, northstar_cluster = self.recluster_selected_points(
                    self.df, points, sl_prop_gradient, north_stars=north_star)

        elif (comp_id == 'bt_north_star' and event_type == 'n_clicks') or \
            (comp_id == 'sl_prop_gradient' and event_type == 'value'):

            figure, northstar_cluster = self.create_graph(
                self.df, gradient_prop=sl_prop_gradient, north_stars=north_star)
        
        elif (comp_id == 'north_star'  and event_type == 'value'):
            north_star = self.update_new_chembl(north_star)
            if north_star:
                figure, northstar_cluster = self.create_graph(
                    self.df, gradient_prop=sl_prop_gradient, north_stars=north_star)
            else:
                raise dash.exceptions.PreventUpdate
        else:
            raise dash.exceptions.PreventUpdate

        return figure, ','.join(northstar_cluster)

    def update_new_chembl(self, north_stars, radius=2, nBits=512):
        north_stars = list(map(str.strip, north_stars.split(',')))
        north_stars = list(map(str.upper, north_stars))
        missing_chembl = set(north_stars).difference(self.chembl_ids)
        
        # CHEMBL10307, CHEMBL103071, CHEMBL103072
        if missing_chembl:
            missing_chembl = list(missing_chembl)
            ldf = self.create_dataframe_molecule_properties(missing_chembl)

            if ldf.shape[0] > 0:
                self.prop_df = self.prop_df.append(ldf)
                self.chembl_ids.extend(missing_chembl)
                
                smiles = []
                for i in range(0, ldf.shape[0]):
                    smiles.append(ldf.iloc[i]['canonical_smiles'].to_array()[0])
                results = list(map(self.MorganFromSmiles, smiles))
                fingerprints = cupy.stack(results).astype(np.float32)
                tdf = self.re_cluster(self.df, fingerprints, missing_chembl)
                if tdf is not None:
                    self.df = tdf
                else:
                    return None
        return ','.join(north_stars)

    def fetch_molecule_properties(self, chemblIDs):
        with closing(sqlite3.connect(CHEMBL_DB)) as con, con,  \
            closing(con.cursor()) as cur:
            select_stmt = '''
                SELECT md.chembl_id, cp.*, cs.*
                FROM compound_properties cp, compound_structures cs, molecule_dictionary md 
                WHERE cp.molregno = md.molregno 
                    AND md.molregno = cs.molregno
                    AND md.chembl_id in (%s);
            ''' % "'%s'" %"','".join(chemblIDs)
            cur.execute(select_stmt)
            cols = list(map(lambda x: x[0], cur.description))
            return cols, cur.fetchall()

    def create_dataframe_molecule_properties(self, chemblIDs):
        with closing(sqlite3.connect(CHEMBL_DB)) as con, con,  \
            closing(con.cursor()) as cur:
            select_stmt = '''
                SELECT md.chembl_id, cp.*, cs.*
                FROM compound_properties cp, molecule_dictionary md, compound_structures cs
                WHERE cp.molregno = md.molregno 
                    AND md.molregno = cs.molregno
                    AND md.chembl_id in (%s);
            ''' % "'%s'" %"','".join(chemblIDs)

            df = cudf.from_pandas(pd.read_sql(select_stmt, con))
            return df.sort_values('chembl_id')
