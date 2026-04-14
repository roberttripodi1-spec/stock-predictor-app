
# Replace your build_simple_gauge function with this custom HTML version,
# and replace the two st.plotly_chart(...) calls in Indicators with st.markdown(...)

def build_simple_gauge_html(title: str, value: float, min_value: float, max_value: float, value_fmt: str) -> str:
    pct = 0.0
    if max_value > min_value:
        pct = (value - min_value) / (max_value - min_value)
    pct = max(0.0, min(1.0, pct))

    # 180 degrees across the top semicircle
    deg = 180 * pct

    if value_fmt == "pct0":
        display_value = f"{value:.0f}"
    elif value_fmt == "float2":
        display_value = f"{value:.2f}"
    else:
        display_value = f"{value}"

    return f"""
    <div style="
        background:#0f172a;
        border:1px solid #223046;
        border-radius:12px;
        padding:12px 12px 8px 12px;
        margin-bottom:10px;
    ">
        <div style="
            color:#93c5fd;
            font-size:0.82rem;
            font-weight:700;
            margin-bottom:6px;
            text-align:center;
        ">{title}</div>

        <div style="
            position:relative;
            width:100%;
            max-width:260px;
            height:150px;
            margin:0 auto;
        ">
            <div style="
                position:absolute;
                left:50%;
                top:12px;
                transform:translateX(-50%);
                width:180px;
                height:90px;
                border-top-left-radius:180px;
                border-top-right-radius:180px;
                border:12px solid #223046;
                border-bottom:0;
                box-sizing:border-box;
                overflow:hidden;
            "></div>

            <div style="
                position:absolute;
                left:50%;
                top:12px;
                transform:translateX(-50%);
                width:180px;
                height:90px;
                border-top-left-radius:180px;
                border-top-right-radius:180px;
                border:12px solid transparent;
                border-bottom:0;
                box-sizing:border-box;
                overflow:hidden;
            ">
                <div style="
                    position:absolute;
                    inset:0;
                    background:conic-gradient(from 180deg, #60a5fa 0deg, #60a5fa {deg}deg, transparent {deg}deg, transparent 180deg);
                    -webkit-mask:
                        radial-gradient(circle at 50% 100%, transparent 52px, black 53px);
                    mask:
                        radial-gradient(circle at 50% 100%, transparent 52px, black 53px);
                "></div>
            </div>

            <div style="
                position:absolute;
                left:50%;
                bottom:26px;
                transform:translateX(-50%);
                color:#f8fafc;
                font-size:1.35rem;
                font-weight:800;
                line-height:1;
                text-align:center;
                width:100%;
            ">{display_value}</div>
        </div>
    </div>
    """

# In your Indicators section, replace the gauge rendering with this:

rsi_value = max(0, min(100, safe_attr(result, "rsi_14", 50.0)))
sentiment_value = max(-1, min(1, safe_attr(result, "sentiment_score", 0.0)))

st.markdown(build_simple_gauge_html("RSI", rsi_value, 0, 100, "pct0"), unsafe_allow_html=True)
st.markdown(build_simple_gauge_html("News Sentiment", sentiment_value, -1, 1, "float2"), unsafe_allow_html=True)
