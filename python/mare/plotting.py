from collections import OrderedDict

import numpy as np
import plotly.graph_objects as go


CHERRY = "rgba(137,28,86,.9)"
TEAL = "rgba(57,117,121,.9)"
ORANGE = "rgba(212,129,59,.9)"
PURPLE = "rgba(136,104,156,.9)"
SAND = "rgba(186,171,155,.9)"

DOMAIN_COLORS = OrderedDict(
    {
        "Health": PURPLE,
        "Other": ORANGE,
        "Energy": TEAL,
        "Safety": CHERRY,
        "Entertainment": SAND,
    }
)

DOMAIN_SYMBOLS = {
    "Health": 0,
    "Other": 1,
    "Energy": 2,
    "Safety": 3,
    "Entertainment": 4,
}


class Plotter(object):
    def __init__(self):
        self.figure = go.Figure()

    def add_trace(self, traces, texts, colors, legend, symbol_number=0):
        self.figure.add_trace(
            go.Scatter(
                x=traces[:, 0],
                y=traces[:, 1],
                mode="markers",
                marker_color=colors,
                text=texts,
                name=legend,
                marker_symbol=DOMAIN_SYMBOLS[legend],
                marker_size=6,
            )
        )

    def add_traces_by_domain(
        self, traces, sentences, colors, domains, filter_domain, legend=None
    ):
        zipped_for_filter = list(zip(traces, sentences, colors, domains))
        domain_specific = list(
            filter(lambda x: x[3] == filter_domain, zipped_for_filter)
        )
        unzipped_for_tracing = list(zip(*domain_specific))
        filtered = {
            "traces": np.array(unzipped_for_tracing[0]),
            "sentences": unzipped_for_tracing[1],
            "colors": unzipped_for_tracing[2],
        }
        self.add_trace(
            filtered["traces"],
            filtered["sentences"],
            filtered["colors"],
            filter_domain if not legend else legend,
        )

    def show(self):
        self.figure.update_layout(
            paper_bgcolor="rgba(255,255,255,0)",
            plot_bgcolor="rgba(255,255,255,0)",
            font=dict(family="serif", color="black", size=18,),
            legend=dict(x=-0.02, orientation="h", font=dict(size=22,)),
        )
        self.figure.update_xaxes(
            showgrid=False,
            gridwidth=1,
            gridcolor="rgba(42, 63, 95,0)",
            zerolinecolor="rgba(42, 63, 95,0)",
        )
        self.figure.update_yaxes(
            showgrid=False,
            gridwidth=1,
            gridcolor="rgba(42, 63, 95,0)",
            zerolinecolor="rgba(42, 63, 95,0)",
        )
        self.figure.show()

    def save(self, filename, extension=".pdf"):
        self.figure.update_layout(showlegend=False)
        self.figure.write_image(filename + extension)
        self.figure.update_layout(showlegend=True)
        self.figure.write_image(filename + "_legend" + extension)


class RequirementsPlotter(object):
    def __init__(self, requirements, colors, domains, legends):
        self.requirements = requirements
        self.colors = colors
        self.domains = domains
        self.legends = legends

    def plot_traces(self, traces):
        p = Plotter()
        for legend in self.legends:
            p.add_traces_by_domain(
                traces, self.requirements, self.colors, self.domains, legend
            )
        p.show()
        return p
