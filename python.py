# I'll create a Python file with the key insights and methods discussed above.
python_code = """
# Python script for space mission data analysis and interactive visualization

import pandas as pd
import plotly.express as px
import folium
import dash
from dash import dcc, html
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from dash.dependencies import Input, Output

# Loading dataset
df = pd.read_csv('spacex_launches.csv')

# Data Wrangling - Handling missing values and outliers
df.fillna(df.mean(), inplace=True)  # Impute missing numerical values
df = df[df['flight_number'] > 0]   # Remove rows with invalid flight numbers

# EDA: Visualization of launch success by site
fig = px.bar(df, x="launch_site", y="flight_number", color="class", title="Launch Success by Site")

# Predictive Analysis: Decision Tree Classifier
X = df[['flight_number', 'launch_site']]  # Features
y = df['class']  # Target variable (success or failure)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision Tree Classifier Model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Predict on test data
y_pred = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, y_pred)

# Random Forest Classifier Model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Predict on test data
y_pred_rf = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, y_pred_rf)

# Folium Map: Display launch sites on an interactive map
map = folium.Map(location=[19.4334, -80.5723], zoom_start=5)
folium.Marker([28.5733, -80.648], popup="Launch Site 1").add_to(map)

# Dash Application for interactive visualization
app = dash.Dash(__name__)

# Layout
app.layout = html.Div([
    html.H1("SpaceX Launch Prediction and Data Visualization"),
    dcc.Graph(figure=fig),
    html.Div([
        html.Label('Select Launch Site:'),
        dcc.Dropdown(
            id='launch-site-dropdown',
            options=[{'label': site, 'value': site} for site in df['launch_site'].unique()],
            value=df['launch_site'].iloc[0]
        ),
        dcc.Graph(id='launch-site-graph')
    ]),
    html.Div([
        html.H3("Model Performance: Decision Tree vs Random Forest"),
        html.P(f"Decision Tree Accuracy: {dt_accuracy*100:.2f}%"),
        html.P(f"Random Forest Accuracy: {rf_accuracy*100:.2f}%")
    ])
])

# Callback to update the graph based on the selected launch site
@app.callback(
    Output('launch-site-graph', 'figure'),
    Input('launch-site-dropdown', 'value')
)
def update_graph(selected_site):
    filtered_df = df[df['launch_site'] == selected_site]
    fig_site = px.scatter(filtered_df, x="flight_number", y="class", title=f"Launch Success for {selected_site}")
    return fig_site

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
"""

# Saving the python code to a file
python_file_path = "/mnt/data/space_mission_analysis.py"

with open(python_file_path, "w") as f:
    f.write(python_code)

python_file_path
