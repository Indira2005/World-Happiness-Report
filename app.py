import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from scipy.stats import ttest_ind
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

st.set_page_config(page_title="World Happiness Analysis", layout="wide")
st.title("World Happiness Analysis Dashboard")
st.markdown("🌍Explore global happiness scores and related factors over time.")

st.markdown("""
    <style>
    div[data-baseweb="select"] > div {
        cursor: pointer !important;
    }
    [data-testid="stExpander"] {
        cursor: pointer !important;
    }
    .stButton > button {
        cursor: pointer !important;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    happiness_df = pd.read_csv('https://raw.githubusercontent.com/UBC-MDS/DSCI-532_2025_11_world_happiness/refs/heads/main/data/processed/World_Happiness_processed_data.csv')
    location_df = pd.read_csv('https://raw.githubusercontent.com/UBC-MDS/DSCI-532_2025_11_world_happiness/refs/heads/main/data/processed/world_countries.csv')
    df = pd.merge(happiness_df, location_df, on='Country', how='inner')
    df = df.drop(columns=['Continent_y'])
    return df

combined_df = load_data()
combined_df = combined_df.drop(columns=['Unnamed: 0'], errors='ignore')
st.sidebar.header("Select View")
view = st.sidebar.selectbox("Choose a section", [
    "Overview",
    "Exploratory Analysis",
    "Regression Analysis",
    "Clustering",
    "PCA",
    "Classification",
    "Forecasting",
    "Statistical Tests"
])

if view == "Overview":
    st.subheader("Data Overview")
    st.dataframe(combined_df.head())

    st.write("### Distribution of Happiness Scores")
    fig, ax = plt.subplots()
    sns.histplot(combined_df['Happiness Score'], kde=True, ax=ax)
    st.pyplot(fig)
    st.markdown("**Requirement:** Visualize the spread and central tendency of happiness scores.")
    st.markdown("**Explanation:** This histogram shows how happiness scores are distributed globally.")
    st.markdown("**Takeaway:** Most countries have moderate scores, with fewer very high or low values.")

elif view == "Exploratory Analysis":
    st.subheader("Regional Differences")
    fig, ax = plt.subplots()
    sns.boxplot(x='Continent_x', y='Happiness Score', data=combined_df, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    st.pyplot(fig)
    st.markdown("**Requirement:** Compare happiness across continents.")
    st.markdown("**Explanation:** Box plots help understand distribution, median, and outliers in each region.")
    st.markdown("**Takeaway:** Europe and North America typically have higher happiness levels than others.")

    st.subheader("Freedom vs Happiness")
    fig, ax = plt.subplots()
    ax.scatter(combined_df['Freedom to Make Life Choices'], combined_df['Happiness Score'], alpha=0.7, color='teal')
    ax.set_xlabel('Freedom to Make Life Choices')
    ax.set_ylabel('Happiness Score')
    st.pyplot(fig)
    st.markdown("**Requirement:** Identify the relationship between freedom and happiness.")
    st.markdown("**Explanation:** This scatter plot helps observe how freedom correlates with happiness.")
    st.markdown("**Takeaway:** Countries with higher freedom often report higher happiness.")

    st.subheader("Average Factor Contributions")
    factor_columns = ['GDP per Capita', 'Social Support', 'Healthy Life Expectancy', 'Freedom to Make Life Choices', 'Perceptions of Corruption', 'Generosity']
    factor_contributions = combined_df[factor_columns].mean().sort_values(ascending=False)
    fig, ax = plt.subplots()
    factor_contributions.plot(kind='bar', color='lightgreen', ax=ax)
    st.pyplot(fig)
    st.markdown("**Requirement:** Understand which factors contribute most to happiness.")
    st.markdown("**Explanation:** Bar chart of average values of happiness-related factors.")
    st.markdown("**Takeaway:** GDP, social support, and life expectancy are key drivers globally.")

    st.subheader("Animated Global Happiness Choropleth")
    fig = px.choropleth(combined_df,
                        locations="Country",
                        locationmode='country names',
                        color="Happiness Score",
                        animation_frame="Year",
                        color_continuous_scale=px.colors.sequential.Plasma)
    st.plotly_chart(fig)
    st.markdown("**Requirement:** Visualize happiness across countries over time.")
    st.markdown("**Explanation:** Animated choropleth map shows dynamic changes in scores annually.")
    st.markdown("**Takeaway:** Happiness trends are stable for most nations, with slight yearly shifts.")

elif view == "Regression Analysis":
    st.subheader("Linear Regression")
    features = ['GDP per Capita', 'Social Support', 'Healthy Life Expectancy', 'Freedom to Make Life Choices', 'Perceptions of Corruption', 'Generosity']
    X = combined_df[features].dropna()
    y = combined_df.loc[X.index, 'Happiness Score']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.4f}")
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, alpha=0.7)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    st.pyplot(fig)
    st.markdown("**Requirement:** Predict happiness using regression.")
    st.markdown("**Explanation:** Compares predicted vs actual values using linear regression.")
    st.markdown("**Takeaway:** The model fits reasonably well, especially in middle-score ranges.")

elif view == "Clustering":
    st.subheader("KMeans Clustering")
    X = combined_df[['GDP per Capita', 'Social Support', 'Freedom to Make Life Choices']].dropna()
    kmeans = KMeans(n_clusters=5, random_state=42)
    combined_df['Cluster'] = kmeans.fit_predict(X)
    fig, ax = plt.subplots()
    sns.scatterplot(data=combined_df, x='GDP per Capita', y='Happiness Score', hue='Cluster', palette='viridis', ax=ax)
    st.pyplot(fig)
    st.markdown("**Requirement:** Group countries based on happiness-related indicators.")
    st.markdown("**Explanation:** KMeans clusters countries based on GDP, support, and freedom.")
    st.markdown("**Takeaway:** Distinct clusters emerge, highlighting varying development patterns.")

elif view == "PCA":
    st.subheader("Principal Component Analysis")
    features = ['GDP per Capita', 'Social Support', 'Healthy Life Expectancy', 'Freedom to Make Life Choices', 'Perceptions of Corruption', 'Generosity']
    df_pca = combined_df.dropna(subset=features)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_pca[features])
    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(data=pca_components, columns=['PC1', 'PC2'])
    pca_df['Continent'] = df_pca['Continent_x'].values
    fig, ax = plt.subplots()
    sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Continent', palette='Set2', ax=ax)
    st.pyplot(fig)
    st.markdown("**Requirement:** Reduce dimensionality to visualize data better.")
    st.markdown("**Explanation:** PCA helps project multi-dimensional data into two main axes.")
    st.markdown("**Takeaway:** Continents show separation in PCA space, reflecting regional patterns.")

elif view == "Classification":
    st.subheader("Random Forest Classification")
    threshold = combined_df['Happiness Score'].quantile(0.75)
    combined_df['TopHappiness'] = (combined_df['Happiness Score'] >= threshold).astype(int)
    features = ['GDP per Capita', 'Social Support', 'Healthy Life Expectancy', 'Freedom to Make Life Choices', 'Perceptions of Corruption', 'Generosity']
    X = combined_df[features].dropna()
    y = combined_df.loc[X.index, 'TopHappiness']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))
    fig, ax = plt.subplots()
    sns.barplot(x=rf.feature_importances_, y=X.columns, palette='viridis', ax=ax)
    ax.set_title('Feature Importance')
    st.pyplot(fig)
    st.markdown("**Requirement:** Classify whether a country is in the top 25% happiest.")
    st.markdown("**Explanation:** Random forest feature importance shows most influential factors.")
    st.markdown("**Takeaway:** GDP, social support, and life expectancy are key for top-tier happiness.")

elif view == "Forecasting":
    st.subheader("Polynomial Forecasting")
    country = st.selectbox("Select a country for forecasting", combined_df['Country'].unique())
    country_data = combined_df[combined_df['Country'] == country]
    happiness_by_year = country_data.groupby('Year')['Happiness Score'].mean().reset_index()
    X = happiness_by_year[['Year']]
    y = happiness_by_year['Happiness Score']
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)
    future_years = np.arange(2025, 2030).reshape(-1, 1)
    future_pred = model.predict(poly.transform(future_years))
    fig, ax = plt.subplots()
    ax.scatter(X, y, label='Actual')
    ax.plot(X, model.predict(X_poly), color='green', label='Polynomial Fit')
    ax.plot(future_years, future_pred, 'r--', label='Forecast')
    ax.set_title(f'Happiness Forecast for {country}')
    ax.set_xlabel('Year')
    ax.set_ylabel('Happiness Score')
    ax.legend()
    st.pyplot(fig)
    st.markdown("**Requirement:** Forecast future happiness scores for a selected country.")
    st.markdown("**Explanation:** Polynomial regression captures nonlinear trends across years.")
    st.markdown("**Takeaway:** The forecast offers a data-driven estimate for upcoming happiness levels.")

elif view == "Statistical Tests":
    st.subheader("T-Test: Developed vs Developing")
    latest_year = combined_df['Year'].max()
    latest_data = combined_df[combined_df['Year'] == latest_year].copy()
    gdp_threshold = latest_data['GDP per Capita'].median()
    latest_data['Development Status'] = latest_data['GDP per Capita'].apply(lambda x: 'Developed' if x > gdp_threshold else 'Developing')
    developed = latest_data[latest_data['Development Status'] == 'Developed']['Happiness Score']
    developing = latest_data[latest_data['Development Status'] == 'Developing']['Happiness Score']
    t_stat, p_val = ttest_ind(developed, developing, equal_var=False)
    st.write(f"T-statistic: {t_stat:.4f}")
    st.write(f"P-value: {p_val:.4f}")
    st.write("Result:", "Statistically significant difference" if p_val < 0.05 else "No significant difference")
    fig, ax = plt.subplots()
    sns.boxplot(data=latest_data, x='Development Status', y='Happiness Score', palette='pastel', ax=ax)
    st.pyplot(fig)
    st.markdown("**Requirement:** Compare happiness between developed and developing countries.")
    st.markdown("**Explanation:** T-test determines if the difference in means is statistically significant.")
    st.markdown("**Takeaway:** Developed countries are generally happier, with statistical confirmation.")

    st.subheader("Tukey HSD: Continent-wise Comparison")
    tukey = pairwise_tukeyhsd(endog=latest_data['Happiness Score'], groups=latest_data['Continent_x'], alpha=0.05)
    st.text(tukey.summary())
    fig = tukey.plot_simultaneous(figsize=(10, 6))
    st.pyplot(fig.figure)
    st.markdown("**Requirement:** Compare continent-level differences in happiness.")
    st.markdown("**Explanation:** Tukey HSD test reveals significant pairwise differences between continents.")
    st.markdown("**Takeaway:** Some regions differ substantially in happiness, e.g., Europe vs Africa.")
