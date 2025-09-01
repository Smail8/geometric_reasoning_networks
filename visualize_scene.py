import argparse
from utils import visualize_scene_from_path
from flask import Flask, render_template_string
import plotly.io as pio


app = Flask(__name__)

# Arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Visualize dataset scenes')

    parser.add_argument('--path', type=str, 
                        help='Scene path',
                        required=True)

    args = parser.parse_args()

    return args

@app.route('/')
def plot():
    # Create a figure using graph_objects
    fig = visualize_scene_from_path(args.path, show=False)

    # Convert the figure to HTML
    graph_html = pio.to_html(fig, full_html=False)

    # Basic HTML template
    html_template = """
    <!doctype html>
    <html lang="en">
      <head><title>Plotly Plot</title></head>
      <body>{{ plot_div|safe }}</body>
    </html>
    """

    return render_template_string(html_template, plot_div=graph_html)

if __name__ == '__main__':
    args = parse_args()
    app.run(host='0.0.0.0', port=8090)