import json

import dash
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc, html
from scipy import stats

PRIOR_PRESETS = {
    "uniform": {"alpha": 1.0, "beta": 1.0, "label": "Uniform  Beta(1, 1)"},
    "jeffreys": {"alpha": 0.5, "beta": 0.5, "label": "Jeffreys  Beta(0.5, 0.5)"},
    "symmetric": {"alpha": 10.0, "beta": 10.0, "label": "Symmetric  Beta(10, 10)"},
    "skeptical": {"alpha": 20.0, "beta": 20.0, "label": "Skeptical  Beta(20, 20)"},
    "heads_skew": {"alpha": 8.0, "beta": 2.0, "label": "Heads-skewed  Beta(8, 2)"},
    "tails_skew": {"alpha": 2.0, "beta": 8.0, "label": "Tails-skewed  Beta(2, 8)"},
}

THETA = np.linspace(0.001, 0.999, 500)

COLORS = {
    "prior": "rgba(136, 153, 180, 0.20)",
    "prior_line": "rgba(136, 153, 180, 0.50)",
    "posterior": "rgba(0, 102, 255, 0.35)",
    "posterior_line": "#0066FF",
    "ghost": "rgba(136, 153, 180, 0.10)",
    "ghost_line": "rgba(136, 153, 180, 0.20)",
    "true_theta": "#FF6B6B",
    "mle": "#FFB366",
    "card_bg": "#0A1024",
    "card_border": "#1C2747",
    "accent": "#0066FF",
    "text_muted": "#8899B4",
    "surface": "#111A33",
}

TOOLTIPS = {
    "preset": "Choose a starting belief about the coin. "
              "'Uniform' means you have no idea if it's fair or biased.",
    "alpha": "How many imaginary heads you've already seen. "
             "Higher means you already believe the coin favors heads.",
    "beta": "How many imaginary tails you've already seen. "
            "Higher means you already believe the coin favors tails.",
    "bias": "The actual hidden probability this coin lands heads. "
            "The red dashed line on the chart shows this truth.",
    "batch": "How many coin flips happen each time you click Flip Coin.",
    "flips": "Total number of coin flips observed so far.",
    "heads": "How many times the coin landed heads.",
    "tails": "How many times the coin landed tails.",
    "mean": "Your best single guess for the coin's bias, "
            "the average of the posterior distribution.",
    "ci": "There is a 95% probability the true coin bias "
          "falls inside this range, given your prior and data.",
    "prob": "The probability the coin is biased toward heads "
            "(lands heads more than half the time).",
    "mode": "The single most likely value of the coin's bias, "
            "the peak of the posterior curve.",
    "var": "How spread out your belief still is. "
           "Smaller means more certain about the bias.",
    "mle": "The frequentist estimate: heads \u00f7 total flips. "
           "Ignores the prior entirely.",
    "kl": "How much your belief changed from the prior. "
          "Zero means no change; larger meansdata shifted your belief a lot.",
}

INFO_STYLE = {
    "display": "inline-flex",
    "alignItems": "center",
    "justifyContent": "center",
    "width": "15px",
    "height": "15px",
    "borderRadius": "50%",
    "backgroundColor": "rgba(0, 102, 255, 0.15)",
    "color": "#0066FF",
    "fontSize": "0.6rem",
    "fontWeight": "700",
    "fontStyle": "italic",
    "fontFamily": "Georgia, serif",
    "cursor": "help",
    "marginLeft": "5px",
    "verticalAlign": "middle",
    "lineHeight": "1",
    "userSelect": "none",
    "flexShrink": "0",
}

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    title="Bayesian Posterior Evolution",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

def _info(tip_id, text):
    icon_id = f"info-{tip_id}"
    return html.Span(
        [
            html.Span("i", id=icon_id, style=INFO_STYLE),
            dbc.Tooltip(
                text,
                target=icon_id,
                placement="right",
                style={"fontSize": "0.8rem", "maxWidth": "260px"},
            ),
        ],
        style={"display": "inline-flex", "alignItems": "center"},
    )


def _stat_card(card_id, label, tip_key, initial="\u2014"):
    return dbc.Card(
        dbc.CardBody(
            [
                html.P(
                    [
                        html.Span(label),
                        _info(f"stat-{tip_key}", TOOLTIPS[tip_key]),
                    ],
                    className="mb-1",
                    style={
                        "color": COLORS["text_muted"],
                        "fontSize": "0.72rem",
                        "letterSpacing": "0.05em",
                        "textTransform": "uppercase",
                        "fontWeight": "500",
                        "display": "flex",
                        "alignItems": "center",
                        "gap": "0",
                    },
                ),
                html.H4(
                    initial,
                    id=card_id,
                    className="mb-0 stat-value",
                    style={
                        "fontWeight": "600",
                        "fontFamily": "'JetBrains Mono', 'Fira Code', monospace",
                        "fontSize": "1.05rem",
                    },
                ),
            ],
            className="py-1 px-3",
        ),
        style={
            "backgroundColor": COLORS["card_bg"],
            "border": f"1px solid {COLORS['card_border']}",
            "borderRadius": "10px",
        },
        className="h-100",
    )


def _slider_with_bounds(slider_id, min_val, max_val, step, value, fmt=".1f"):
    return html.Div(
        [
            dcc.Slider(
                id=slider_id,
                min=min_val,
                max=max_val,
                step=step,
                value=value,
                marks=None,
                tooltip={"placement": "bottom", "always_visible": True},
            ),
            html.Div(
                [
                    html.Span(f"{min_val:{fmt}}", style={"color": COLORS["text_muted"], "fontSize": "0.7rem"}),
                    html.Span(f"{max_val:{fmt}}", style={"color": COLORS["text_muted"], "fontSize": "0.7rem"}),
                ],
                style={"display": "flex", "justifyContent": "space-between", "marginTop": "-4px", "paddingInline": "6px"},
            ),
        ]
    )


def _control_group(label_text, child, tip_key=None, margin_bottom="1.1rem"):
    label_children = [html.Span(label_text)]
    if tip_key:
        label_children.append(_info(f"ctrl-{tip_key}", TOOLTIPS[tip_key]))

    return html.Div(
        [
            html.Label(
                label_children,
                style={
                    "color": COLORS["text_muted"],
                    "fontSize": "0.82rem",
                    "fontWeight": "500",
                    "marginBottom": "0.35rem",
                    "display": "flex",
                    "alignItems": "center",
                },
            ),
            child,
        ],
        style={"marginBottom": margin_bottom},
    )


def _sidebar():
    return dbc.Card(
        dbc.CardBody(
            [
                html.H5(
                    "Controls",
                    className="mb-4",
                    style={"fontWeight": "700", "letterSpacing": "0.03em"},
                ),

                _control_group(
                    "Prior Preset",
                    dcc.Dropdown(
                        id="preset-dropdown",
                        options=[{"label": v["label"], "value": k} for k, v in PRIOR_PRESETS.items()],
                        value="uniform",
                        clearable=False,
                    ),
                    tip_key="preset",
                    margin_bottom="1.3rem",
                ),

                _control_group(
                    "Prior \u03b1",
                    _slider_with_bounds("alpha-slider", 0.1, 50, 0.1, 1.0),
                    tip_key="alpha",
                ),

                _control_group(
                    "Prior \u03b2",
                    _slider_with_bounds("beta-slider", 0.1, 50, 0.1, 1.0),
                    tip_key="beta",
                ),

                html.Hr(style={"borderColor": COLORS["card_border"], "margin": "0.6rem 0 1rem 0"}),

                _control_group(
                    "True Coin Bias (\u03b8*)",
                    _slider_with_bounds("bias-slider", 0.01, 0.99, 0.01, 0.6, fmt=".2f"),
                    tip_key="bias",
                ),

                _control_group(
                    "Flips per Click",
                    dcc.Dropdown(
                        id="batch-dropdown",
                        options=[{"label": str(n), "value": n} for n in [1, 5, 10, 25, 50]],
                        value=1,
                        clearable=False,
                    ),
                    tip_key="batch",
                    margin_bottom="1.5rem",
                ),

                dbc.Button(
                    "Flip Coin",
                    id="flip-btn",
                    color="primary",
                    className="w-100 mb-2 btn-flip",
                    style={
                        "fontWeight": "600",
                        "letterSpacing": "0.04em",
                        "borderRadius": "8px",
                        "fontSize": "1.05rem",
                        "padding": "0.6rem",
                    },
                ),
                dbc.Button(
                    "Reset",
                    id="reset-btn",
                    outline=True,
                    color="secondary",
                    className="w-100 btn-reset",
                    style={"borderRadius": "8px", "padding": "0.5rem"},
                ),
            ],
            className="px-3 py-3",
        ),
        style={
            "backgroundColor": COLORS["card_bg"],
            "border": f"1px solid {COLORS['card_border']}",
            "borderRadius": "12px",
            "flex": "1",
        },
    )


# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------

app.layout = html.Div(
    [
        dcc.Store(id="session-store", data=json.dumps({"heads": 0, "tails": 0, "history": [], "flip_results": []})),

        html.Div(
            [
                html.H5(
                    "Bayesian Posterior Evolution",
                    style={"fontWeight": "700", "letterSpacing": "-0.01em", "marginBottom": "0.1rem", "fontSize": "1.25rem"},
                ),
                html.P(
                    "Watch a Beta posterior update in real time as evidence accumulates.",
                    style={"color": COLORS["text_muted"], "fontSize": "0.85rem", "marginBottom": 0},
                ),
            ],
            style={"paddingBottom": "0.6rem"},
        ),

        dbc.Row(
            [
                dbc.Col(_sidebar(), width=12, lg=3, className="d-flex"),
                dbc.Col(
                    [
                        dbc.Card(
                            dbc.CardBody(
                                dcc.Graph(id="posterior-graph", config={"displayModeBar": False, "responsive": True}),
                            ),
                            style={
                                "backgroundColor": COLORS["card_bg"],
                                "border": f"1px solid {COLORS['card_border']}",
                                "borderRadius": "12px",
                            },
                            className="mb-2 chart-card",
                        ),

                        dbc.Row(
                            [
                                dbc.Col(_stat_card("stat-flips", "Total Flips", "flips"), xs=6, md=4, lg=True),
                                dbc.Col(_stat_card("stat-heads", "Heads", "heads"), xs=6, md=4, lg=True),
                                dbc.Col(_stat_card("stat-tails", "Tails", "tails"), xs=6, md=4, lg=True),
                                dbc.Col(_stat_card("stat-mean", "Post. Mean", "mean"), xs=6, md=4, lg=True),
                                dbc.Col(_stat_card("stat-ci", "95% CI", "ci"), xs=6, md=4, lg=True),
                            ],
                            className="g-2 mb-2",
                        ),
                        dbc.Row(
                            [
                                dbc.Col(_stat_card("stat-prob", "P(\u03b8 > 0.5)", "prob"), xs=6, md=3, lg=True),
                                dbc.Col(_stat_card("stat-mode", "Post. Mode", "mode"), xs=6, md=3, lg=True),
                                dbc.Col(_stat_card("stat-var", "Post. Variance", "var"), xs=6, md=3, lg=True),
                                dbc.Col(_stat_card("stat-mle", "MLE (\u03b8\u0302)", "mle"), xs=6, md=3, lg=True),
                                dbc.Col(_stat_card("stat-kl", "KL(Post \u2016 Prior)", "kl"), xs=6, md=3, lg=True),
                            ],
                            className="g-2",
                        ),
                    ],
                    width=12,
                    lg=9,
                    className="right-col",
                ),
            ],
            className="main-row align-items-stretch",
        ),
    ],
    className="app-container",
)


# ---------------------------------------------------------------------------
# Visualization builder
# ---------------------------------------------------------------------------

def _build_figure(alpha_prior, beta_prior, heads, tails, history, true_bias):
    a_post = alpha_prior + heads
    b_post = beta_prior + tails

    fig = go.Figure()

    prior_y = stats.beta.pdf(THETA, alpha_prior, beta_prior)
    fig.add_trace(go.Scatter(
        x=THETA, y=prior_y,
        fill="tozeroy",
        fillcolor=COLORS["prior"],
        line=dict(color=COLORS["prior_line"], width=1.5),
        name="Prior",
        hoverinfo="skip",
    ))

    for i, (a_h, b_h) in enumerate(history):
        if i % max(1, len(history) // 8) != 0:
            continue
        ghost_y = stats.beta.pdf(THETA, a_h, b_h)
        fig.add_trace(go.Scatter(
            x=THETA, y=ghost_y,
            fill="tozeroy",
            fillcolor=COLORS["ghost"],
            line=dict(color=COLORS["ghost_line"], width=1, dash="dot"),
            showlegend=False,
            hoverinfo="skip",
        ))

    n_obs = heads + tails
    if n_obs > 0:
        post_y = stats.beta.pdf(THETA, a_post, b_post)
        fig.add_trace(go.Scatter(
            x=THETA, y=post_y,
            fill="tozeroy",
            fillcolor=COLORS["posterior"],
            line=dict(color=COLORS["posterior_line"], width=2.5),
            name=f"Posterior  Beta({a_post:.1f}, {b_post:.1f})",
        ))

    post_peak = stats.beta.pdf(THETA, a_post, b_post).max() if n_obs > 0 else 0
    plot_ymax = max(prior_y.max(), post_peak) * 1.15

    fig.add_trace(go.Scatter(
        x=[true_bias, true_bias],
        y=[0, plot_ymax],
        mode="lines",
        line=dict(color=COLORS["true_theta"], width=2, dash="dash"),
        name=f"\u03b8* = {true_bias:.2f}",
        hoverinfo="skip",
    ))

    if n_obs > 0:
        mle = heads / n_obs
        fig.add_trace(go.Scatter(
            x=[mle],
            y=[0],
            mode="markers",
            marker=dict(symbol="triangle-up", size=12, color=COLORS["mle"]),
            name=f"MLE = {mle:.3f}",
        ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=58, r=20, t=28, b=44),
        xaxis=dict(
            title="\u03b8 (coin bias)",
            range=[0, 1],
            gridcolor="rgba(255,255,255,0.04)",
            tickvals=[0.2, 0.4, 0.6, 0.8, 1.0],
            showline=True,
            linecolor="rgba(255,255,255,0.08)",
            ticklen=12,
            tickcolor="rgba(0,0,0,0)",
        ),
        yaxis=dict(
            title="Density",
            rangemode="tozero",
            gridcolor="rgba(255,255,255,0.04)",
            showline=True,
            linecolor="rgba(255,255,255,0.08)",
            ticklen=8,
            tickcolor="rgba(0,0,0,0)",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=11)),
        transition={"duration": 350, "easing": "cubic-in-out"},
        font=dict(family="'Inter', sans-serif"),
    )

    return fig


def _compute_stats(alpha_prior, beta_prior, heads, tails):
    n = heads + tails
    a_post = alpha_prior + heads
    b_post = beta_prior + tails

    post_mean = a_post / (a_post + b_post)
    post_var = (a_post * b_post) / ((a_post + b_post) ** 2 * (a_post + b_post + 1))

    if a_post > 1 and b_post > 1:
        post_mode = (a_post - 1) / (a_post + b_post - 2)
    else:
        post_mode = None

    ci_lo, ci_hi = stats.beta.ppf([0.025, 0.975], a_post, b_post)
    prob_gt_half = 1 - stats.beta.cdf(0.5, a_post, b_post)
    mle = heads / n if n > 0 else None

    kl = _kl_beta(a_post, b_post, alpha_prior, beta_prior)

    return {
        "n": n,
        "heads": heads,
        "tails": tails,
        "mean": post_mean,
        "mode": post_mode,
        "var": post_var,
        "ci": (ci_lo, ci_hi),
        "prob_gt_half": prob_gt_half,
        "mle": mle,
        "kl": kl,
    }


def _kl_beta(a1, b1, a2, b2):
    """KL divergence D_KL(Beta(a1,b1) || Beta(a2,b2))."""
    from scipy.special import betaln, digamma
    kl = (betaln(a2, b2) - betaln(a1, b1)
          + (a1 - a2) * digamma(a1)
          + (b1 - b2) * digamma(b1)
          + (a2 - a1 + b2 - b1) * digamma(a1 + b1))
    return max(float(kl), 0.0)


def _fmt(val, precision=4):
    if val is None:
        return "\u2014"
    return f"{val:.{precision}f}"


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

DEFAULTS = {"preset": "uniform", "alpha": 1.0, "beta": 1.0, "bias": 0.6, "batch": 1}


@callback(
    Output("alpha-slider", "value"),
    Output("beta-slider", "value"),
    Input("preset-dropdown", "value"),
    prevent_initial_call=True,
)
def apply_preset(preset_key):
    p = PRIOR_PRESETS[preset_key]
    return p["alpha"], p["beta"]


@callback(
    Output("preset-dropdown", "value"),
    Output("bias-slider", "value"),
    Output("batch-dropdown", "value"),
    Input("reset-btn", "n_clicks"),
    prevent_initial_call=True,
)
def reset_controls(_n):
    return DEFAULTS["preset"], DEFAULTS["bias"], DEFAULTS["batch"]


@callback(
    Output("posterior-graph", "figure"),
    Output("session-store", "data"),
    Output("stat-flips", "children"),
    Output("stat-heads", "children"),
    Output("stat-tails", "children"),
    Output("stat-mean", "children"),
    Output("stat-ci", "children"),
    Output("stat-prob", "children"),
    Output("stat-mode", "children"),
    Output("stat-var", "children"),
    Output("stat-mle", "children"),
    Output("stat-kl", "children"),
    Input("flip-btn", "n_clicks"),
    Input("reset-btn", "n_clicks"),
    Input("alpha-slider", "value"),
    Input("beta-slider", "value"),
    State("bias-slider", "value"),
    State("batch-dropdown", "value"),
    State("session-store", "data"),
)
def update_dashboard(flip_clicks, reset_clicks, alpha, beta_param, true_bias, batch_size, store_json):
    ctx = dash.callback_context
    triggered = ctx.triggered[0]["prop_id"] if ctx.triggered else ""

    store = json.loads(store_json)

    if "reset-btn" in triggered or "alpha-slider" in triggered or "beta-slider" in triggered:
        store = {"heads": 0, "tails": 0, "history": [], "flip_results": []}

    elif "flip-btn" in triggered:
        flips = np.random.binomial(1, true_bias, size=batch_size)
        new_heads = int(flips.sum())
        new_tails = batch_size - new_heads
        store["heads"] += new_heads
        store["tails"] += new_tails
        store["flip_results"].extend(flips.tolist())
        store["history"].append([alpha + store["heads"], beta_param + store["tails"]])

    fig = _build_figure(alpha, beta_param, store["heads"], store["tails"], store["history"], true_bias)
    s = _compute_stats(alpha, beta_param, store["heads"], store["tails"])

    return (
        fig,
        json.dumps(store),
        str(s["n"]),
        str(s["heads"]),
        str(s["tails"]),
        _fmt(s["mean"]),
        f"[{s['ci'][0]:.3f}, {s['ci'][1]:.3f}]",
        _fmt(s["prob_gt_half"], 3),
        _fmt(s["mode"]),
        _fmt(s["var"], 6),
        _fmt(s["mle"]),
        _fmt(s["kl"], 4),
    )


if __name__ == "__main__":
    app.run(debug=True, port=8050)
