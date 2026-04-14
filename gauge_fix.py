
# Replace ONLY your build_simple_gauge function with this

def build_simple_gauge(title: str, value: float, min_value: float, max_value: float):
    import plotly.graph_objects as go

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={
            "font": {"size": 28},
            "valueformat": ".2f"
        },
        gauge={
            "axis": {"range": [min_value, max_value]},
            "bar": {"color": "#60a5fa"},
            "bgcolor": "#111827",
            "bordercolor": "#223046",
        },
        title={"text": ""}  # remove top title so it doesn't shift layout
    ))

    fig.update_layout(
        height=150,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e5edf8"),
    )

    return fig
