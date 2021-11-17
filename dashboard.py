import os

import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import plotly.express as px

import pandas as pd
import numpy as np
import json

from plyfile import PlyData
from sklearn import preprocessing

DIR = "/home/thzou/Projects/torch-points3d"
benchmark_dir = os.path.join(DIR, 'outputs/benchmark/')
print(benchmark_dir)

def normalize(df):
    x = df.values  # returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    dfo = pd.DataFrame(x_scaled)
    dfo.index = df.index
    dfo.columns = df.columns
    return dfo


def fimiou(a, imiou, total_time, gpu_mem):
    fimiou_v = a * imiou + (1 - a) / 2 * (1 - total_time) + (1 - a) / 2 * (1 - gpu_mem)
    return fimiou_v, a * imiou, (1 - a) / 2 * (1 - total_time), (1 - a) / 2 * (1 - gpu_mem)


def fcmiou(b, cmiou, total_time, gpu_mem):
    fcmiou_v = b * cmiou + (1 - b) / 2 * (1 - total_time) + (1 - b) / 2 * (1 - gpu_mem)
    return fcmiou_v, b * cmiou, (1 - b) / 2 * (1 - total_time), (1 - b) / 2 * (1 - gpu_mem)


def fgeneral(fimiou_v, fcmiou_v):
    fgeneral_v = (fimiou_v + fcmiou_v) / 2
    return fgeneral_v, fimiou_v / 2, fcmiou_v / 2


def calculations_util(a, b, df):
    data = []
    for index in df.index:
        dfimiou = fimiou(a, df['imiou'][index], df['time'][index], df['gpu_mem'][index])
        dfcmiou = fcmiou(b, df['cmiou'][index], df['time'][index], df['gpu_mem'][index])
        dfgeneral = fgeneral(dfimiou[0], dfcmiou[0])
        data.append([dfimiou, dfcmiou, dfgeneral])

    # Create a dataframe for results
    columns_ids = ['Fimiou', 'Fcmiou', 'Fgeneral']
    index_rows = df.index

    df = pd.DataFrame(columns=columns_ids, index=index_rows, data=data)

    return df


class PlyVisualization(object):
    def __init__(self, file):
        self.plydata = PlyData.read(file)
        columns = ['x', 'y', 'z', 'l', 'p']
        self.df = pd.DataFrame({col: self.plydata['data_visual'][col] for col in columns})
        for col in ['l', 'p']:
            self.df[col] = self.df[col].astype(int).astype(str)
        self.df['error'] = self.df['l'] != self.df['p']
        self.df.rename({'l': 'ground-truth', 'p': 'prediction'}, axis=1, inplace=True)
        minimum_value = int(self.df[['ground-truth', 'prediction']].min().min())
        maximum_value = int(self.df[['ground-truth', 'prediction']].max().max())
        self.colors_dict = {str(i): px.colors.qualitative.Dark24[i % 24] for i in
                            range(minimum_value, maximum_value + 1)}

    def plot(self, **kwargs):
        fig = self.fig(**kwargs)
        fig.show()

    def fig(self, **kwargs):
        if 'color' in kwargs:
            if kwargs['color'] == 'error':
                fig = px.scatter_3d(self.df, x='z', y='x', z='y', color_discrete_map={True: 'red', False: 'green'},
                                    **kwargs)
            else:
                fig = px.scatter_3d(self.df, x='z', y='x', z='y', color_discrete_map=self.colors_dict, **kwargs)
        else:
            fig = px.scatter_3d(self.df, x='z', y='x', z='y', color_discrete_map=self.colors_dict, **kwargs)
        fig.update_traces(marker=dict(size=1))
        return fig

    def metric(self, metric):
        hypothesis_test = {c: {
            'TP': ((self.df['prediction'] == c) & (self.df['ground-truth'] == self.df['prediction'])).sum(),
            'FP': ((self.df['prediction'] == c) & (self.df['ground-truth'] != self.df['prediction'])).sum(),
            'TN': ((self.df['prediction'] != c) & (self.df['ground-truth'] == self.df['prediction'])).sum(),
            'FN': ((self.df['prediction'] != c) & (self.df['ground-truth'] != self.df['prediction'])).sum(),
        } for c in self.df['ground-truth'].unique()}
        if metric == 'recall':
            return {c: hypothesis_test[c]['TP'] / (hypothesis_test[c]['TP'] + hypothesis_test[c]['FN'])
            if (hypothesis_test[c]['TP'] + hypothesis_test[c]['FN']) != 0 else 0
                    for c in self.df['ground-truth'].unique()}
        elif metric == 'precision':
            return {c: hypothesis_test[c]['TP'] / (hypothesis_test[c]['TP'] + hypothesis_test[c]['FP'])
            if (hypothesis_test[c]['TP'] + hypothesis_test[c]['FP']) != 0 else 0
                    for c in self.df['ground-truth'].unique()}
        elif metric == 'accuracy':
            return {c: (hypothesis_test[c]['TP'] + hypothesis_test[c]['TN']) / (
                    hypothesis_test[c]['TP'] + hypothesis_test[c]['TN'] + hypothesis_test[c]['FP'] +
                    hypothesis_test[c]['FN']) for c in self.df['ground-truth'].unique()}
        elif metric == 'f1-score':
            return {c: hypothesis_test[c]['TP'] / (
                    hypothesis_test[c]['TP'] + 0.5 * (hypothesis_test[c]['FP'] + hypothesis_test[c]['FN']))
                    for c in self.df['ground-truth'].unique()}
        else:
            raise ValueError("Unknown metric")


class Experiment(object):
    METRICS = ['recall', 'precision', 'accuracy', 'f1-score']

    def __init__(self, root_folder: str):
        self.root_folder = root_folder
        self.viz_folder = os.path.join(os.path.join(self.root_folder, 'viz'))
        self.epochs = sorted([int(epoch) for epoch in os.listdir(self.viz_folder)])

    def get_classes(self):
        with open(os.path.join(self.root_folder, 'train.log'), 'r') as file:
            file_data = file.read()
        first_epoch_data = file_data.split('EPOCH ')[1]

        class_imiou_first_epoch = [json.loads(line.split('=')[1].replace("'", '"'))
                                   for line in first_epoch_data.split('\n')
                                   if line.strip().startswith('test_Imiou_per_class')]
        return class_imiou_first_epoch[0].keys()

    def load_epoch_files(self, set_index: int, n_epoch: int = 1):
        if n_epoch not in self.epochs:
            raise IndexError(f'Invalid epoch {n_epoch} for experiment {self.root_folder}')
        epoch_folder = os.path.join(self.viz_folder, str(n_epoch))
        val_files, test_files, train_files = [
            [os.path.join(os.path.join(epoch_folder, folder), subfolder) for subfolder in
             os.listdir(os.path.join(epoch_folder, folder))] for folder in os.listdir(epoch_folder)]
        return [val_files, test_files, train_files][set_index]

    def load_log_wandb(self):
        wandb_dir = os.path.join(self.root_folder, 'wandb')
        run_dir = os.path.join(wandb_dir, [d for d in os.listdir(wandb_dir) if 'run' in d][0])
        log_file = os.path.join(run_dir, 'wandb-history.jsonl')
        with open(log_file, 'r') as f:
            log = [json.loads(line) for line in f.readlines()]
        return pd.DataFrame.from_records(log)

    def load_train_file(self):
        with open(os.path.join(self.root_folder, 'train.log'), 'r') as file:
            file_data = file.read()
        epochs_data = file_data.split('EPOCH ')[1:]
        class_imiou_per_epoch = [json.loads(line.split('=')[1].replace("'", '"'))
                                 for epoch_data in epochs_data
                                 for line in epoch_data.split('\n')
                                 if line.strip().startswith('test_Imiou_per_class')
                                 ]
        return class_imiou_per_epoch

    def fig_for_log(self):
        log = self.load_log_wandb()
        figs = {}
        kinds = ['Cmiou', 'Imiou', 'loss_seg']
        cols_by_kind = [[col for col in log if col.endswith(k)] for k in kinds]
        for k, cols in zip(kinds, cols_by_kind):
            fig = go.Figure()
            for col in cols:
                fig.add_trace(go.Scatter(x=log['_step'], y=log[col], mode='lines+markers', name=col))
                fig.update_yaxes(type="log", range=(1.69, 2))
            figs[k] = fig
        return figs

    def figs_for_epoch(self, set_index: int = 0, n_epoch: int = 1, sample_index: int = 0):
        files = self.load_epoch_files(set_index, n_epoch)
        visualizer = PlyVisualization(files[sample_index])
        return visualizer.fig(color='ground-truth'), visualizer.fig(color='prediction'), visualizer.fig(color='error')

    def metric(self, set_index: int = 0, sample_index: int = 0, metric: str = METRICS[0]):
        metrics = {}
        for epoch in range(1, 200):
            files = self.load_epoch_files(set_index, epoch)
            epoch_dic = {}
            file = files[sample_index]
            epoch_dic[file] = PlyVisualization(file).metric(metric)
            metrics[epoch] = epoch_dic
        df = pd.DataFrame(
            [(epoch, file, str(c), rec) for epoch, _ in metrics.items() for file, __ in _.items() for c, rec in
             __.items()],
            columns=['Epoch', 'File', 'Feature', metric.capitalize()])
        return df.pivot_table(index=['Epoch', 'Feature'], values=metric.capitalize(), aggfunc=np.mean).reset_index()

experiments = os.listdir(benchmark_dir)
roots = [os.path.join(benchmark_dir, exp) for exp in experiments]
exps = [Experiment(root) for root in roots]

pointnet = [84.24, 79.03, 5.55, 33.7]
pointnet2 = [84.93, 82.50, 4.55, 36.3]
kpconv = [84.22, 82.39, 23.01, 60.4]
ppnet = [83.87, 82.50, 29.28, 89.0]
rsconv = [85.47, 82.73, 12.06, 55.5]
columns_ids = ['imiou', 'cmiou', 'time', 'gpu_mem']
index_rows = ['PointNet', 'PointNet++', 'KPConv', 'PPNet', 'RSConv']
default_data = pd.DataFrame(columns=columns_ids, index=index_rows, data=[pointnet, pointnet2, kpconv, ppnet, rsconv])

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

terms = {
    'Fimiou': {1: 'a*imiou', 2: '(1-a)/2 * (1-total_time)', 3: '(1-a)/2 * (1-gpu_mem)'},
    'Fcmiou': {1: 'b*cmiou', 2: '(1-b)/2 * (1-total_time)', 3: '(1-b)/2 * (1-gpu_mem)'},
    'Fgeneral': {1: 'Fimiou_v/2', 2: 'Fcmiou_v/2'}
}

colors = {
    'Fcmiou': {1: '#1abc9c', 2: '#76d7c4', 3: '#a3e4d7'},
    'Fimiou': {1: '#3498db', 2: '#85c1e9', 3: '#aed6f1'},
    'Fgeneral': {1: '#c0392b', 2: '#cd6155'}
}

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    dcc.Tabs([
        dcc.Tab(label="General metrics", children=[
            html.Div([
                html.H3("Comparison"),
                dcc.Store(id='default-models', data=default_data.to_records()),
                dcc.Loading(type='default', children=dcc.Graph(id='graph-comparison')),
                html.H3("Data"),
                dcc.Loading(type='default', children=dash_table.DataTable(id='data-table',
                                                                          columns=[{"name": i, "id": i} for i in
                                                                                   ['name', 'imiou', 'cmiou', 'time',
                                                                                    'gpu_mem']]
                                                                          ))
            ], className="eight columns"),
            html.Div([
                html.H3("Parameters"),
                html.P('Alpha:'),
                html.Div(id='alpha-slider-value'),
                dcc.Slider(id='alpha-slider', min=0, max=1, step=0.01, value=0.5),
                html.P('Beta:'),
                html.Div(id='beta-slider-value'),
                dcc.Slider(id='beta-slider', min=0, max=1, step=0.01, value=0.5),
                html.P("Select Models:"),
                dcc.Dropdown(id='selected-models-comparison',
                             options=[
                                 {'label': i, 'value': i} for i in default_data.index
                             ],
                             value=default_data.index,
                             multi=True
                             ),
                html.Div(
                    [html.H3("Add Model")] +
                    [html.P(f'{i}:') if j == 0
                     else dcc.Input(id=f'input_model_{i}', type=t, placeholder=i) if j == 1
                     else html.Br() if j == 2
                     else html.Br()
                     for t, i in zip(["text", "number", "number", "number", "number"],
                                     ['name', 'imiou', 'cmiou', 'time', 'gpu_mem'])
                     for j in range(4)] +
                    [html.Button('Add Model', id='add-model-btn')]
                )
            ], className="four columns")
        ]),
        dcc.Tab(label="Per class analysis", children=[
            html.Div([
                html.Div([
                    html.H3("Options"),
                    html.Br(),
                    html.Br(),
                    html.Br(),
                    html.Br(),
                    html.P("Select Models:"),
                    dcc.Dropdown(id='comparison-models',
                                 options=[
                                     {'label': exp.root_folder.split('-')[-2], 'value': exp.root_folder} for exp in exps
                                 ],
                                 value=[exp.root_folder for exp in exps],
                                 multi=True
                                 ),
                    html.Br(),
                    html.P("Select Sets:"),
                    dcc.Dropdown(id='comparison-sets',
                                 options=[
                                     {'label': s, 'value': s} for s in ['test', 'train']
                                 ],
                                 value=['test', 'train'],
                                 multi=True
                                 ),
                    html.Br(),
                    html.P("Select Classes:"),
                    dcc.Dropdown(id='comparison-classes',
                                 options=[
                                     {'label': i, 'value': i} for i in []
                                 ],
                                 value=[],
                                 multi=True
                                 )
                ], className="six columns"),
                html.Div([
                    html.H3("CmIoU"),
                    dcc.Loading(type='default', children=dcc.Graph(id='comparison-graph-CmIoU'))
                ], className="six columns")
            ], className="row"),
            html.Div([
                html.Div([
                    html.H3("mIoU per class - Best CmIoU"),
                    dcc.Loading(type='default', children=dcc.Graph(id='comparison-graph-class-mIoU'))
                ], className="six columns"),
                html.Div([
                    html.H3("ImIoU"),
                    dcc.Loading(type='default', children=dcc.Graph(id='comparison-graph-ImIoU'))
                ], className="six columns")
            ], className="row")
        ]),
        dcc.Tab(label="Model analysis", children=[
            html.Div([
                html.Div([
                    html.H3('Benchmark'),
                    html.Br(),
                    html.P('Select Current Experiment:'),
                    dcc.RadioItems(
                        id='experiment-selector',
                        options=[
                            {'label': exp.root_folder.split('-')[-2], 'value': exp.root_folder} for exp in exps
                        ],
                        value=exps[0].root_folder
                    )
                ], className="six columns"),
                html.Div([
                    html.H3('CmIoU'),
                    dcc.Loading(type='default', children=dcc.Graph(id='experiment-graph-CmIoU'))
                ], className="six columns")
            ], className="row"),
            html.Div([
                html.Div([
                    html.H3('mIoU per Class'),
                    dcc.Loading(type='default', children=dcc.Graph(id='experiment-graph-class-mIoU'))
                ], className="six columns"),
                html.Div([
                    html.H3('ImIoU'),
                    dcc.Loading(type='default', children=dcc.Graph(id='experiment-graph-ImIoU'))
                ], className="six columns")
            ], className="row")
        ]),
        dcc.Tab(label="Visual inspection", children=[
            html.Div([
                html.Div([
                    html.H3('Sample Plot'),
                    html.Br(),
                    html.Br(),
                    dcc.Dropdown(id='set-dropdown'),
                    html.Br(),
                    dcc.Slider(id='epoch-slider'),
                    html.P(id='epoch-label'),
                    dcc.Dropdown(id='sample-dropdown'),
                    html.H3('Metric'),
                    dcc.Dropdown(
                        id='metric-dropdown',
                        options=[
                            {'label': metric, 'value': metric} for metric in Experiment.METRICS
                        ],
                        value="recall")
                ], className="four columns"),
                html.Div([
                    html.H4('Per feature metrics in sample'),
                    dcc.Loading(type='default', children=dcc.Graph(id='experiment-graph-Metric'))
                ], className="eight columns")
            ], className="row"),
            html.Div([
                html.Div([
                    html.H4('Ground Truth'),
                    dcc.Loading(type='default', children=dcc.Graph(id='ground-truth'))
                ], className="four columns"),
                html.Div([
                    html.H4('Prediction'),
                    dcc.Loading(type='default', children=dcc.Graph(id='prediction'))
                ], className="four columns"),
                html.Div([
                    html.H4('Error'),
                    dcc.Loading(type='default', children=dcc.Graph(id='error'))
                ], className="four columns")
            ], className="row")
        ])
    ])
])


###############
#   General   #
###############
@app.callback(
    Output('graph-comparison', 'figure'),
    Input('default-models', 'data'),
    Input('alpha-slider', 'value'),
    Input('beta-slider', 'value'),
    Input('selected-models-comparison', 'value')
)
def update_comparison_graph(data, alpha, beta, models):
    df = pd.DataFrame.from_records(data)
    df.index = df.iloc[:, 0].values
    df.drop(0, axis=1, inplace=True)
    df.columns = ['imiou', 'cmiou', 'time', 'gpu_mem']
    df = df.loc[models, :]
    ndf = normalize(df)
    calcs = calculations_util(alpha, beta, ndf)
    fig = go.Figure()

    fig.update_layout(
        xaxis=dict(title_text="Model"),
        yaxis=dict(title_text="Score"),
        barmode="stack",
    )

    for metric in ['Fcmiou', 'Fimiou', 'Fgeneral']:
        if metric == 'Fgeneral':
            n = 3
        else:
            n = 4
        for i in range(1, n):
            fig.add_trace(
                go.Bar(x=[calcs.index, [metric] * len(calcs)], y=[cal[i] for cal in calcs[metric]],
                       marker_color=colors[metric][i], name=terms[metric][i])
            )
    return fig


@app.callback(
    Output('alpha-slider-value', 'children'),
    Input('alpha-slider', 'value')
)
def update_alpha_slider(alpha):
    return str(alpha)


@app.callback(
    Output('beta-slider-value', 'children'),
    Input('beta-slider', 'value')
)
def update_beta_slider(beta):
    return str(beta)


@app.callback(
    Output('data-table', 'data'),
    Input('default-models', 'data')
)
def update_data_table(data):
    df = pd.DataFrame.from_records(data)
    df.index = df.iloc[:, 0].values
    df.columns = ['name', 'imiou', 'cmiou', 'time', 'gpu_mem']
    return df.to_dict('records')


@app.callback(
    Output('selected-models-comparison', 'options'),
    Output('selected-models-comparison', 'value'),
    Input('default-models', 'data')
)
def update_models_options(data):
    models = pd.DataFrame.from_records(data)[0].values
    return [{'label': i, 'value': i} for i in models], models


@app.callback(
    Output('default-models', 'data'),
    Input('add-model-btn', 'n_clicks'),
    State('default-models', 'data'),
    State('input_model_name', 'value'),
    State('input_model_imiou', 'value'),
    State('input_model_cmiou', 'value'),
    State('input_model_time', 'value'),
    State('input_model_gpu_mem', 'value')
)
def add_model(n_clicks, data, name, imiou, cmiou, time, gpu_mem):
    if n_clicks is None:
        return data
    else:
        df = pd.DataFrame.from_records(data).append({0: name, 1: imiou, 2: cmiou, 3: time, 4: gpu_mem},
                                                    ignore_index=True)
        return df.to_records(index=False)


###############
# Comparison  #
###############
@app.callback(
    Output('comparison-classes', 'options'),
    Output('comparison-classes', 'value'),
    Input('comparison-models', 'value')
)
def update_classes_options(models):
    if models:
        classes = set([cl for root_folder in models for cl in Experiment(root_folder).get_classes()])
        sorted_classes = sorted(classes)
        return [{'label': cl, 'value': cl} for cl in sorted_classes], sorted_classes
    else:
        return [], []

@app.callback(
    Output('comparison-graph-CmIoU', 'figure'),
    Output('comparison-graph-ImIoU', 'figure'),
    Input('comparison-models', 'value'),
    Input('comparison-sets', 'value')
)
def update_cmiou_imiou_comparison_graphs(models, sets):
    if models:
        logs = []
        for exp_folder in models:
            experiment = Experiment(exp_folder)
            logs.append(experiment.load_log_wandb())

        figs = {}
        for model, log in zip(models, logs):
            model_name = model.split('-')[-2]
            kinds = ['Cmiou', 'Imiou']
            cols_by_kind = [[col for col in log if col.endswith(k)] for k in kinds]
            for k, cols in zip(kinds, cols_by_kind):
                if k not in figs:
                    fig = go.Figure()
                    fig.update_yaxes(type="log", range=(1.69, 2))
                else:
                    fig = figs[k]
                for col in cols:
                    if any(s in col for s in sets):
                        fig.add_trace(go.Scatter(x=log['_step'], y=log[col], mode='lines+markers', name=model_name+"-"+col))
                figs[k] = fig

        return figs['Cmiou'], figs['Imiou']
    else:
        return go.Figure(), go.Figure()

@app.callback(
    Output('comparison-graph-class-mIoU', 'figure'),
    Input('comparison-models', 'value'),
    Input('comparison-classes', 'value')
)
def update_class_miou_comparison_graphs(models, classes):
    if models:
        best_imious = pd.DataFrame()
        for exp_folder in models:
            experiment = Experiment(exp_folder)
            class_imiou_per_epoch = experiment.load_train_file()
            df = pd.DataFrame.from_records(class_imiou_per_epoch)
            best_imious[exp_folder.split('-')[-2]] = df.max(axis=0)

        plot_ready_df = best_imious.loc[classes].reset_index().rename({'index': 'Class'}, axis=1).melt(
            id_vars='Class', var_name='Model', value_name='mIoU', ignore_index=True)
        fig = px.line(plot_ready_df, x='Class', y='mIoU', color='Model')
        fig.update_yaxes(range=(0.5, 1))
        return fig
    else:
        return go.Figure()

###############
#    Model    #
###############
@app.callback(
    Output('set-dropdown', 'options'),
    Output('set-dropdown', 'value'),
    Input('experiment-selector', 'value')
)
def update_set_options(exp_folder):
    return [{'label': s, 'value': i} for i, s in enumerate(['validation', 'test', 'train'])], 0


@app.callback(
    Output('epoch-slider', 'value'),
    Output('epoch-slider', 'min'),
    Output('epoch-slider', 'max'),
    Output('epoch-slider', 'step'),
    Input('experiment-selector', 'value')
)
def update_epoch(exp_folder):
    experiment = Experiment(exp_folder)
    minimum = min(experiment.epochs)
    maximum = max(experiment.epochs)
    return minimum, minimum, maximum, 1


@app.callback(
    Output('epoch-label', 'children'),
    Input('epoch-slider', 'value')
)
def update_epoch_label(epoch):
    return f'Selected epoch: {epoch}'


@app.callback(
    Output('sample-dropdown', 'options'),
    Output('sample-dropdown', 'value'),
    Input('epoch-slider', 'value'),
    State('set-dropdown', 'value'),
    State('experiment-selector', 'value')
)
def update_sample_options(epoch, set_index, exp_folder):
    experiment = Experiment(exp_folder)
    files = experiment.load_epoch_files(set_index, epoch)
    return [{'label': s.split('/')[-1], 'value': i} for i, s in enumerate(files)], 0


@app.callback(
    Output('experiment-graph-Metric', 'figure'),
    Input('experiment-selector', 'value'),
    Input('set-dropdown', 'value'),
    Input('sample-dropdown', 'value'),
    Input('metric-dropdown', 'value')
)
def update_experiment_metric(exp_folder, set_index, sample_index, metric):
    experiment = Experiment(exp_folder)
    a = experiment.metric(set_index, sample_index, metric)
    colors_dict = {str(i): px.colors.qualitative.Dark24[int(i) % 24] for i in
                   range(0, a['Feature'].astype(int).max() + 1)}
    fig = px.line(a, x='Epoch', y=metric.capitalize(), color='Feature', color_discrete_map=colors_dict)
    fig.update_yaxes(range=(0, 1))
    return fig


@app.callback(
    Output('experiment-graph-CmIoU', 'figure'),
    Output('experiment-graph-ImIoU', 'figure'),
    Input('experiment-selector', 'value')
)
def update_experiment_log_figures(exp_folder):
    experiment = Experiment(exp_folder)
    mfig = experiment.fig_for_log()
    return mfig['Cmiou'], mfig['Imiou']


@app.callback(
    Output('experiment-graph-class-mIoU', 'figure'),
    Input('experiment-selector', 'value')
)
def update_experiment_per_class_imiou(exp_folder):
    experiment = Experiment(exp_folder)
    class_imiou_per_epoch = experiment.load_train_file()
    df = pd.DataFrame.from_records(class_imiou_per_epoch)
    classes = df.columns
    df['Epoch'] = range(1, df.shape[0] + 1)
    plot_ready_df = df.melt(id_vars='Epoch', value_vars=classes, var_name='Class', value_name='mIoU',
                            ignore_index=True)
    fig = px.line(plot_ready_df, x='Epoch', y='mIoU', color='Class')
    fig.update_yaxes(range=(0, 1))
    return fig


@app.callback(
    Output('ground-truth', 'figure'),
    Output('prediction', 'figure'),
    Output('error', 'figure'),
    Input('sample-dropdown', 'value'),
    State('set-dropdown', 'value'),
    State('epoch-slider', 'value'),
    State('experiment-selector', 'value')
)
def update_sample_figures(sample_index, set_index, epoch, exp_folder):
    experiment = Experiment(exp_folder)
    figs = experiment.figs_for_epoch(set_index=set_index, n_epoch=epoch, sample_index=sample_index)
    return figs[0], figs[1], figs[2]


if __name__ == "__main__":
    app.run_server(
        debug=True, port=8051, dev_tools_hot_reload=False, use_reloader=False
    )
