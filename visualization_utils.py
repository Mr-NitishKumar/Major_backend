import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

def create_pair_plot(df, output_file="static/pair_plot.png"):
    """Generate a pair plot for numeric features."""
    numeric_columns = df.select_dtypes(include=['number']).columns
    sns.pairplot(df[numeric_columns])
    plt.savefig(output_file)
    plt.close()
    return output_file

def create_box_plots(df, output_dir="static/box_plots"):
    """Generate box plots for numeric features."""
    numeric_columns = df.select_dtypes(include=['number']).columns
    box_plots = {}
    for col in numeric_columns:
        plt.figure()
        sns.boxplot(y=df[col])
        filename = f"{output_dir}/{col}_box_plot.png"
        plt.savefig(filename)
        box_plots[col] = filename
        plt.close()
    return box_plots

def create_interactive_scatter(df, x_col, y_col):
    """Generate an interactive scatter plot using Plotly."""
    fig = px.scatter(df, x=x_col, y=y_col, title=f"Scatter Plot of {x_col} vs {y_col}")
    filename = f"static/{x_col}_vs_{y_col}_scatter.html"
    fig.write_html(filename)
    return filename


def create_clustering_plot(X, clusters, filename="static/cluster_plot.png"):
    """Generate a scatter plot for clustering results."""
    plt.figure(figsize=(8, 6))
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=clusters, cmap='viridis')
    plt.title("Clustering Results")
    plt.xlabel(X.columns[0])
    plt.ylabel(X.columns[1])
    plt.savefig(filename)
    return filename
