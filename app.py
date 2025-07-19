# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np

# Import onze custom modules
from models.price_predictor import VastgoedPrijsPredictor
from models.portfolio_optimizer import VastgoedPortfolioOptimizer
from utils.data_collector import VastgoedDataCollector

# Configuratie
st.set_page_config(
    page_title="Vastgoed Portfolio Optimizer",
    page_icon="🏘️",
    layout="wide"
)

# Titel en introductie
st.title("🏘️ Vastgoed Portfolio Optimizer met AI")
st.markdown("""
Deze applicatie helpt je bij het analyseren en optimaliseren van vastgoedinvesteringen 
met behulp van machine learning en portfolio theorie.
""")

# Sidebar voor navigatie
st.sidebar.title("Navigatie")
page = st.sidebar.radio(
    "Kies een functie:",
    ["🏠 Home", "📊 Markt Analyse", "🤖 Prijs Voorspelling", "💼 Portfolio Optimalisatie"]
)

# Cache data loading voor performance
@st.cache_data
def load_data():
    """Laad vastgoeddata (cached voor snelheid)"""
    try:
        df = pd.read_csv('data/vastgoed_data.csv')
        df['datum'] = pd.to_datetime(df['datum'])
    except:
        st.info("Genereer sample data...")
        df = create_sample_vastgoed_data(5000)
        df.to_csv('data/vastgoed_data.csv', index=False)
    return df

# Load data
df = load_data()

if page == "🏠 Home":
    st.header("Welkom bij de Vastgoed Portfolio Optimizer!")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Totaal Panden", f"{len(df):,}")
    with col2:
        st.metric("Gem. Prijs", f"€{df['prijs'].mean():,.0f}")
    with col3:
        st.metric("Data Periode", f"{df['datum'].min().date()} - {df['datum'].max().date()}")
    
    st.subheader("📈 Recente Prijsontwikkeling")
    
    # Prijstrend per stad
    price_trend = df.groupby(['datum', 'stad'])['prijs'].mean().reset_index()
    
    fig = px.line(
        price_trend, 
        x='datum', 
        y='prijs', 
        color='stad',
        title='Gemiddelde Woningprijs per Stad',
        labels={'prijs': 'Gemiddelde Prijs (€)', 'datum': 'Datum'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("🎯 Wat kan deze app?")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **📊 Markt Analyse**
        - Bekijk actuele markttrends
        - Analyseer prijsontwikkelingen per regio
        - Vergelijk verschillende woningtypes
        """)
        
        st.markdown("""
        **🤖 Prijs Voorspelling**
        - Voorspel woningprijzen met AI
        - Krijg inzicht in prijsbepalende factoren
        - Ontvang confidence intervals
        """)
    
    with col2:
        st.markdown("""
        **💼 Portfolio Optimalisatie**
        - Optimaliseer je vastgoedportfolio
        - Minimaliseer risico bij gewenst rendement
        - Visualiseer de efficient frontier
        """)
        
        st.markdown("""
        **🔍 Data Insights**
        - Ontdek verborgen patronen
        - Identificeer investeringskansen
        - Export rapporten en analyses
        """)

elif page == "📊 Markt Analyse":
    st.header("📊 Markt Analyse Dashboard")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_cities = st.multiselect(
            "Selecteer steden:",
            options=df['stad'].unique(),
            default=df['stad'].unique()
        )
    
    with col2:
        selected_types = st.multiselect(
            "Selecteer woningtypes:",
            options=df['type_woning'].unique(),
            default=df['type_woning'].unique()
        )
    
    with col3:
        date_range = st.date_input(
            "Selecteer periode:",
            value=(df['datum'].min(), df['datum'].max()),
            min_value=df['datum'].min(),
            max_value=df['datum'].max()
        )
    
    # Filter data
    filtered_df = df[
        (df['stad'].isin(selected_cities)) &
        (df['type_woning'].isin(selected_types)) &
        (df['datum'] >= pd.to_datetime(date_range[0])) &
        (df['datum'] <= pd.to_datetime(date_range[1]))
    ]
    
    # Visualisaties
    tab1, tab2, tab3, tab4 = st.tabs(["Prijsverdeling", "Trends", "Correlaties", "Top Performers"])
    
    with tab1:
        st.subheader("Prijsverdeling per Stad")
        fig = px.box(
            filtered_df, 
            x='stad', 
            y='prijs', 
            color='type_woning',
            title='Prijsverdeling per Stad en Woningtype'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistieken tabel
        stats_df = filtered_df.groupby(['stad', 'type_woning'])['prijs'].agg([
            'count', 'mean', 'median', 'std'
        ]).round(0)
        stats_df.columns = ['Aantal', 'Gemiddeld', 'Mediaan', 'Std Dev']
        st.dataframe(stats_df, use_container_width=True)
    
    with tab2:
        st.subheader("Prijstrends over Tijd")
        
        # Maandelijkse gemiddelden
        monthly_avg = filtered_df.groupby([
            pd.Grouper(key='datum', freq='M'), 
            'stad'
        ])['prijs'].mean().reset_index()
        
        fig = px.line(
            monthly_avg,
            x='datum',
            y='prijs',
            color='stad',
            title='Maandelijkse Gemiddelde Prijzen',
            markers=True
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Jaar-over-jaar groei
        st.subheader("Jaar-over-Jaar Prijsgroei (%)")
        yearly_growth = filtered_df.groupby([
            filtered_df['datum'].dt.year,
            'stad'
        ])['prijs'].mean().pct_change(periods=1).reset_index()
        yearly_growth['prijs'] = yearly_growth['prijs'] * 100
        
        fig = px.bar(
            yearly_growth.dropna(),
            x='datum',
            y='prijs',
            color='stad',
            title='Jaarlijkse Prijsgroei Percentage',
            labels={'prijs': 'Groei (%)'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Correlatie Analyse")
        
        # Scatter plot matrix
        numerical_cols = ['oppervlakte', 'kamers', 'bouwjaar', 'prijs']
        fig = px.scatter_matrix(
            filtered_df[numerical_cols].sample(min(1000, len(filtered_df))),
            dimensions=numerical_cols,
            color=filtered_df['prijs'],
            title='Feature Correlaties'
        )
        fig.update_traces(diagonal_visible=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation heatmap
        corr_matrix = filtered_df[numerical_cols].corr()
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            title='Correlatie Heatmap',
            color_continuous_scale='RdBu',
            aspect='auto'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Top Performing Segmenten")
        
        # Beste ROI segmenten
        segment_performance = filtered_df.groupby(['stad', 'type_woning']).agg({
            'prijs': ['mean', 'std', 'count']
        }).round(0)
        
        segment_performance.columns = ['Gem_Prijs', 'Volatiliteit', 'Aantal']
        segment_performance['Sharpe_Ratio'] = (
            segment_performance['Gem_Prijs'] / segment_performance['Volatiliteit']
        ).round(2)
        
        # Top 10 segmenten
        top_segments = segment_performance.sort_values('Sharpe_Ratio', ascending=False).head(10)
        
        st.dataframe(
            top_segments.style.highlight_max(axis=0),
            use_container_width=True
        )
        
        # Visualisatie
        fig = px.scatter(
            segment_performance.reset_index(),
            x='Volatiliteit',
            y='Gem_Prijs',
            size='Aantal',
            color='Sharpe_Ratio',
            hover_data=['stad', 'type_woning'],
            title='Risk-Return Profile per Segment',
            labels={'Gem_Prijs': 'Gemiddelde Prijs (€)', 'Volatiliteit': 'Risico (Std Dev)'}
        )
        st.plotly_chart(fig, use_container_width=True)

elif page == "🤖 Prijs Voorspelling":
    st.header("🤖 AI Prijs Voorspelling")
    
    # Train model als nog niet gebeurd
    if 'price_predictor' not in st.session_state:
        with st.spinner('Model wordt getraind...'):
            predictor = VastgoedPrijsPredictor()
            train_results = predictor.train(df)
            st.session_state.price_predictor = predictor
            st.session_state.train_results = train_results
    
    predictor = st.session_state.price_predictor
    train_results = st.session_state.train_results
    
    # Model performance metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model Accuraatheid (R²)", f"{train_results['r2']:.3f}")
    with col2:
        st.metric("Gemiddelde Afwijking", f"€{train_results['mae']:,.0f}")
    with col3:
        st.metric("Training Samples", f"{len(df):,}")
    
    # Feature importance
    st.subheader("🎯 Belangrijkste Prijsfactoren")
    fig = px.bar(
        train_results['feature_importance'],
        x='importance',
        y='feature',
        orientation='h',
        title='Feature Importance voor Prijsvoorspelling'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Voorspelling interface
    st.subheader("🏠 Voorspel Woningprijs")
    
    col1, col2 = st.columns(2)
    
    with col1:
        stad = st.selectbox("Stad", df['stad'].unique())
        wijk = st.selectbox("Wijk", df['wijk'].unique())
        type_woning = st.selectbox("Type Woning", df['type_woning'].unique())
        energielabel = st.selectbox("Energielabel", sorted(df['energielabel'].unique()))
    
    with col2:
        oppervlakte = st.slider("Oppervlakte (m²)", 40, 300, 100)
        kamers = st.slider("Aantal Kamers", 1, 6, 3)
        bouwjaar = st.slider("Bouwjaar", 1900, 2024, 2000)
    
    if st.button("🔮 Voorspel Prijs", type="primary"):
        # Maak voorspelling
        property_data = {
            'stad': stad,
            'wijk': wijk,
            'type_woning': type_woning,
            'oppervlakte': oppervlakte,
            'kamers': kamers,
            'bouwjaar': bouwjaar,
            'energielabel': energielabel,
            'prijs': 0  # Dummy voor feature engineering
        }
        
        prediction = predictor.predict(property_data)
        
        # Toon resultaten
        st.success("✅ Voorspelling Compleet!")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Voorspelde Prijs",
                f"€{prediction['predicted_price']:,.0f}"
            )
        
        with col2:
            st.metric(
                "Prijs per m²",
                f"€{prediction['price_per_m2']:,.0f}"
            )
        
        with col3:
            st.metric(
                "90% Confidence Interval",
                f"€{prediction['confidence_interval'][0]:,.0f} - €{prediction['confidence_interval'][1]:,.0f}"
            )
        
        # Vergelijk met vergelijkbare woningen
        st.subheader("📊 Vergelijking met Vergelijkbare Woningen")
        
        similar_properties = df[
            (df['stad'] == stad) &
            (df['type_woning'] == type_woning) &
            (df['oppervlakte'].between(oppervlakte - 20, oppervlakte + 20))
        ].head(20)
        
        if len(similar_properties) > 0:
            fig = px.scatter(
                similar_properties,
                x='oppervlakte',
                y='prijs',
                color='energielabel',
                size='kamers',
                hover_data=['wijk', 'bouwjaar'],
                title=f'Vergelijkbare Woningen in {stad}'
            )
            
            # Voeg voorspelling toe
            fig.add_trace(
                go.Scatter(
                    x=[oppervlakte],
                    y=[prediction['predicted_price']],
                    mode='markers',
                    marker=dict(size=20, color='red', symbol='star'),
                    name='Jouw Voorspelling'
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)

elif page == "💼 Portfolio Optimalisatie":
    st.header("💼 Portfolio Optimalisatie")
    
    st.markdown("""
    Gebruik Modern Portfolio Theory om de optimale mix van vastgoedinvesteringen te vinden
    op basis van historisch rendement en risico.
    """)
    
    # Portfolio samenstelling
    st.subheader("🏘️ Selecteer Portfolio Assets")
    
    # Groepeer per stad/type voor portfolio assets
    portfolio_assets = df.groupby(['stad', 'type_woning'])['prijs'].agg(['mean', 'count']).reset_index()
    portfolio_assets = portfolio_assets[portfolio_assets['count'] >= 30]  # Alleen segmenten met voldoende data
    portfolio_assets['asset_name'] = portfolio_assets['stad'] + ' - ' + portfolio_assets['type_woning']
    
    selected_assets = st.multiselect(
        "Kies vastgoed segmenten voor je portfolio:",
        options=portfolio_assets['asset_name'].tolist(),
        default=portfolio_assets['asset_name'].tolist()[:5]
    )
    
    if len(selected_assets) < 2:
        st.warning("Selecteer minimaal 2 assets voor portfolio optimalisatie")
    else:
        # Bereken returns voor geselecteerde assets
        returns_data = []
        
        for asset in selected_assets:
            stad, type_woning = asset.split(' - ')
            asset_prices = df[
                (df['stad'] == stad) & 
                (df['type_woning'] == type_woning)
            ].groupby('datum')['prijs'].mean()
            returns_data.append(asset_prices)
        
        # Combineer returns
        returns_df = pd.DataFrame(returns_data).T
        returns_df.columns = selected_assets
        returns_df = returns_df.pct_change().dropna()
        
        # Initialiseer optimizer
        optimizer = VastgoedPortfolioOptimizer()
        
        # Optimaliseer portfolio
        optimal_weights, optimal_metrics = optimizer.optimize_portfolio(returns_df)
        
        # Toon resultaten
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Optimale Portfolio Verdeling")
            
            # Pie chart van weights
            weights_df = pd.DataFrame({
                'Asset': selected_assets,
                'Gewicht': optimal_weights * 100
            })
            
            fig = px.pie(
                weights_df,
                values='Gewicht',
                names='Asset',
                title='Optimale Asset Allocatie (%)'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Weights tabel
            st.dataframe(
                weights_df.style.format({'Gewicht': '{:.1f}%'}),
                use_container_width=True
            )
        
        with col2:
            st.subheader("📈 Portfolio Metrics")
            
            st.metric("Verwacht Jaarrendement", f"{optimal_metrics['return']*100:.2f}%")
            st.metric("Volatiliteit (Risico)", f"{optimal_metrics['volatility']*100:.2f}%")
            st.metric("Sharpe Ratio", f"{optimal_metrics['sharpe_ratio']:.3f}")
            
            st.info("""
            **Interpretatie:**
            - **Rendement**: Verwacht jaarlijks rendement op basis van historische data
            - **Volatiliteit**: Standaarddeviatie van rendementen (maat voor risico)
            - **Sharpe Ratio**: Rendement per eenheid risico (hoger = beter)
            """)
        
        # Efficient Frontier
        st.subheader("🎯 Efficient Frontier")
        
        with st.spinner("Bereken efficient frontier..."):
            frontier_returns, frontier_volatilities = optimizer.generate_efficient_frontier(returns_df)
        
        # Plot efficient frontier
        fig = go.Figure()
        
        # Frontier lijn
        fig.add_trace(go.Scatter(
            x=frontier_volatilities,
            y=frontier_returns,
            mode='lines',
            name='Efficient Frontier',
            line=dict(color='blue', width=2)
        ))
        
        # Optimale portfolio punt
        fig.add_trace(go.Scatter(
            x=[optimal_metrics['volatility']],
            y=[optimal_metrics['return']],
            mode='markers',
            name='Optimale Portfolio',
            marker=dict(size=15, color='red', symbol='star')
        ))
        
        # Random portfolios voor vergelijking
        n_random = 1000
        random_weights = np.random.dirichlet(np.ones(len(selected_assets)), n_random)
        random_returns = []
        random_volatilities = []
        
        for weights in random_weights:
            metrics = optimizer.calculate_portfolio_metrics(weights, returns_df)
            random_returns.append(metrics['return'])
            random_volatilities.append(metrics['volatility'])
        
        fig.add_trace(go.Scatter(
            x=random_volatilities,
            y=random_returns,
            mode='markers',
            name='Random Portfolios',
            marker=dict(size=3, color='lightgray', opacity=0.5)
        ))
        
        fig.update_layout(
            title='Efficient Frontier: Risk vs Return',
            xaxis_title='Volatiliteit (Risico)',
            yaxis_title='Verwacht Rendement',
            xaxis=dict(tickformat='.1%'),
            yaxis=dict(tickformat='.1%'),
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Portfolio simulatie
        st.subheader("💰 Portfolio Waarde Simulatie")
        
        initial_investment = st.number_input(
            "Initiële Investering (€)",
            min_value=10000,
            max_value=10000000,
            value=100000,
            step=10000
        )
        
        years = st.slider("Projectie Periode (jaren)", 1, 30, 10)
        
        # Monte Carlo simulatie
        n_simulations = 1000
        final_values = []
        
        for _ in range(n_simulations):
            value = initial_investment
            for year in range(years):
                annual_return = np.random.normal(
                    optimal_metrics['return'],
                    optimal_metrics['volatility']
                )
                value *= (1 + annual_return)
            final_values.append(value)
        
        # Resultaten
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Mediaan Eindwaarde",
                f"€{np.median(final_values):,.0f}"
            )
        
        with col2:
            st.metric(
                "5% Percentiel (Worst Case)",
                f"€{np.percentile(final_values, 5):,.0f}"
            )
        
        with col3:
            st.metric(
                "95% Percentiel (Best Case)",
                f"€{np.percentile(final_values, 95):,.0f}"
            )
        
        # Histogram van eindwaardes
        fig = px.histogram(
            final_values,
            nbins=50,
            title=f'Distributie van Portfolio Waarde na {years} jaar',
            labels={'value': 'Portfolio Waarde (€)', 'count': 'Frequentie'}
        )
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
💡 **Tip**: Deze applicatie is gebouwd als onderdeel van een portfolio project voor de transitie naar AI/Data rollen.
Voor vragen of suggesties, neem contact op!
""")