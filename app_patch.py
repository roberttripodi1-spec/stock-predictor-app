
# NOTE: Replace ONLY your movers section with this

if not movers.empty:
    mover_cols = st.columns(5)

    for idx, (_, row) in enumerate(movers.iterrows()):
        ticker = row["Ticker"]
        pct = float(row["Change %"])

        label = f"{ticker} {'▲' if pct >= 0 else '▼'} {pct:+.2f}%"

        with mover_cols[idx % 5]:
            if st.button(label, key=f"mover_{ticker}", use_container_width=True):
                st.session_state.active_ticker = ticker
                st.session_state.auto_run = True
                st.rerun()

            # Styling
            if pct >= 0:
                st.markdown(f'''
                <style>
                button[data-testid="baseButton-secondary"][key="mover_{ticker}"] {{
                    background-color: #0b3b2e !important;
                    color: #86efac !important;
                    border: 1px solid #14532d !important;
                }}
                </style>
                ''', unsafe_allow_html=True)
            else:
                st.markdown(f'''
                <style>
                button[data-testid="baseButton-secondary"][key="mover_{ticker}"] {{
                    background-color: #4c1717 !important;
                    color: #fca5a5 !important;
                    border: 1px solid #7f1d1d !important;
                }}
                </style>
                ''', unsafe_allow_html=True)
else:
    st.caption("Top movers unavailable.")
