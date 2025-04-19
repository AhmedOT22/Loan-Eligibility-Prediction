import plotly.graph_objects as go

def generate_gauge_chart(probability, interpretation, color):
    """
    Generates a Plotly gauge chart visualizing loan approval probability as a percentage.
    
    The chart displays the probability value with a percentage suffix, highlights the current value with a thick pointer, and divides the gauge into color-coded segments representing probability ranges. The chart title includes the provided interpretation and formatted probability value.
    
    Args:
        probability: Numeric value (0–100) representing the loan approval probability.
        interpretation: Text describing the meaning of the probability.
    
    Returns:
        A Plotly Figure object containing the customized gauge chart.
    """
    return go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability,
        number={"suffix": "%", "font": {"size": 36}},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkgray"},
            'bar': {'color': "rgba(0,0,0,0)"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 20], 'color': '#ff4d4d'},
                {'range': [20, 40], 'color': '#ffa64d'},
                {'range': [40, 60], 'color': '#ffff66'},
                {'range': [60, 80], 'color': '#b3ff66'},
                {'range': [80, 100], 'color': '#4dff88'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 6},
                'thickness': 0.9,
                'value': probability
            }
        },
        title={
            'text': f"<b>{interpretation}</b><br>Probability: {probability:.2f}%",
            'font': {'size': 20}
        }
    ))