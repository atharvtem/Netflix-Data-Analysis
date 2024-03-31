import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import seaborn as sns
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.colors


# Load the dataset
def load_data():
    # Assuming the CSV file is in the same directory as this script
    file_path = "netflix.csv"
    data = pd.read_csv(file_path)
    return data

# Handle missing data and replacements
def handle_missing_data(df):
    # Missing data
    for i in df.columns:
        null_rate = df[i].isna().sum() / len(df) * 100 
        if null_rate > 0 :
            st.write("{} null rate: {}%".format(i,round(null_rate,2)))

    # Replacements
    df['country'] = df['country'].fillna(df['country'].mode()[0])
    df['cast'].replace(np.nan, 'No Data', inplace=True)
    df['director'].replace(np.nan, 'No Data', inplace=True)

    # Drops
    df.dropna(inplace=True)

    # Drop Duplicates
    df.drop_duplicates(inplace=True)

    return df

# Additional feature extraction
def add_date_features(df):
    df["date_added"] = pd.to_datetime(df['date_added'], errors='coerce', format='%B %d, %Y')
    df['month_added'] = df['date_added'].dt.month
    df['month_name_added'] = df['date_added'].dt.month_name()
    df['year_added'] = df['date_added'].dt.year
    return df

# Visualize the ratio of movies and TV shows
def visualize_ratio(df):
    x = df.groupby(['type'])['type'].count()
    y = len(df)
    r = ((x/y)).round(2)

    mf_ratio = pd.DataFrame(r).T

    fig, ax = plt.subplots(1,1,figsize=(6.5, 2.5))

    ax.barh(mf_ratio.index, mf_ratio['Movie'], 
            color='#b20710', alpha=0.9, label='Movie')
    ax.barh(mf_ratio.index, mf_ratio['TV Show'], left=mf_ratio['Movie'], 
            color='#221f1f', alpha=0.9, label='TV Show')

    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])

    # Movie percentage
    for i in mf_ratio.index:
        ax.annotate(f"{int(mf_ratio['Movie'][i]*100)}%", 
                       xy=(mf_ratio['Movie'][i]/2, i),
                       va='center', ha='center', fontsize=40, fontweight='light', fontfamily='serif',
                       color='white')

    # TV Show percentage
    for i in mf_ratio.index:
        ax.annotate(f"{int(mf_ratio['TV Show'][i]*100)}%", 
                       xy=(mf_ratio['Movie'][i]+mf_ratio['TV Show'][i]/2, i),
                       va='center', ha='center', fontsize=40, fontweight='light', fontfamily='serif',
                       color='white')

    # Title & Subtitle
    fig.text(0.125,1.03,'Movie & TV Show distribution', fontfamily='serif',fontsize=15, fontweight='bold')
    fig.text(0.125,0.92,'We see vastly more movies than TV shows on Netflix.',fontfamily='serif',fontsize=12)  

    for s in ['top', 'left', 'right', 'bottom']:
        ax.spines[s].set_visible(False)

    # Removing legend due to labelled plot
    ax.legend().set_visible(False)

    # Show the plot
    st.pyplot(fig)

# Main function to run the Streamlit app
    
def quick_feature_engineering(df):
    df['count'] = 1
    df['first_country'] = df['country'].apply(lambda x: x.split(',')[0] if isinstance(x, str) else None)
    ratings_ages = {
        'TV-PG': 'Older Kids',
        'TV-MA': 'Adults',
        'TV-Y7-FV': 'Older Kids',
        'TV-Y7': 'Older Kids',
        'TV-14': 'Teens',
        'R': 'Adults',
        'TV-Y': 'Kids',
        'NR': 'Adults',
        'PG-13': 'Teens',
        'TV-G': 'Kids',
        'PG': 'Older Kids',
        'G': 'Kids',
        'UR': 'Adults',
        'NC-17': 'Adults'
    }
    df['target_ages'] = df['rating'].replace(ratings_ages)
    df['genre'] = df['listed_in'].apply(lambda x: x.replace(' ,',',').replace(', ',',').split(',') if isinstance(x, str) else [])
    df['first_country'].replace('United States', 'USA', inplace=True)
    df['first_country'].replace('United Kingdom', 'UK', inplace=True)
    df['first_country'].replace('South Korea', 'S. Korea', inplace=True)
    return df


# Plot top 10 countries
def plot_top_countries(df):
    data = df.groupby('first_country')['count'].sum().sort_values(ascending=False)[:10]

    color_map = ['#f5f5f1' for _ in range(10)]
    color_map[0] = color_map[1] = color_map[2] =  '#b20710' # color highlight

    fig, ax = plt.subplots(1,1, figsize=(12, 6))
    ax.bar(data.index, data, width=0.5, edgecolor='darkgray', linewidth=0.6, color=color_map)

    for i in data.index:
        ax.annotate(f"{data[i]}", 
                    xy=(i, data[i] + 150), 
                    va='center', ha='center',fontweight='light', fontfamily='serif')

    for s in ['top', 'left', 'right']:
        ax.spines[s].set_visible(False)

    ax.set_xticklabels(data.index, fontfamily='serif', rotation=0)

    fig.text(0.09, 1, 'Top 10 countries on Netflix', fontsize=15, fontweight='bold', fontfamily='serif')
    fig.text(0.09, 0.95, 'The three most frequent countries have been highlighted.', fontsize=12, fontweight='light', fontfamily='serif')

    fig.text(1.1, 1.01, 'Insight', fontsize=15, fontweight='bold', fontfamily='serif')
    fig.text(1.1, 0.67, '''
    The most prolific producers of
    content for Netflix are, primarily,
    the USA, with India and the UK
    a significant distance behind.

    It makes sense that the USA produces 
    the most content as, after all, 
    Netflix is a US company.
    '''
             , fontsize=12, fontweight='light', fontfamily='serif')

    ax.grid(axis='y', linestyle='-', alpha=0.4)   
    grid_y_ticks = np.arange(0, 4000, 500) 
    ax.set_yticks(grid_y_ticks)
    ax.set_axisbelow(True)

    plt.axhline(y = 0, color = 'black', linewidth = 1.3, alpha = .7)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis=u'both', which=u'both',length=0)
    
    st.pyplot(fig)


def handle_missing_data(df):
    # Handle missing data and replacements here
    return df

# Additional feature engineering function
def additional_feature_engineering(df):
    # Perform additional feature engineering here
    return df

# Plot function for top 10 countries movie & TV show split
def plot_country_movie_tv_ratio(df):
    country_order = df['first_country'].value_counts()[:11].index
    data_q2q3 = df[['type', 'first_country']].groupby('first_country')['type'].value_counts().unstack().loc[country_order]
    data_q2q3['sum'] = data_q2q3.sum(axis=1)
    data_q2q3_ratio = (data_q2q3.T / data_q2q3['sum']).T[['Movie', 'TV Show']].sort_values(by='Movie',ascending=False)[::-1]

    fig, ax = plt.subplots(1,1,figsize=(15, 8),)
    ax.barh(data_q2q3_ratio.index, data_q2q3_ratio['Movie'], color='#b20710', alpha=0.8, label='Movie')
    ax.barh(data_q2q3_ratio.index, data_q2q3_ratio['TV Show'], left=data_q2q3_ratio['Movie'], color='#221f1f', alpha=0.8, label='TV Show')

    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.set_yticklabels(data_q2q3_ratio.index, fontfamily='serif', fontsize=11)

    for i in data_q2q3_ratio.index:
        ax.annotate(f"{data_q2q3_ratio['Movie'][i]*100:.3}%", xy=(data_q2q3_ratio['Movie'][i]/2, i), va='center', ha='center', fontsize=12, fontweight='light', fontfamily='serif', color='white')

    for i in data_q2q3_ratio.index:
        ax.annotate(f"{data_q2q3_ratio['TV Show'][i]*100:.3}%", xy=(data_q2q3_ratio['Movie'][i]+data_q2q3_ratio['TV Show'][i]/2, i), va='center', ha='center', fontsize=12, fontweight='light', fontfamily='serif', color='white')

    fig.text(0.13, 0.93, 'Top 10 countries Movie & TV Show split', fontsize=15, fontweight='bold', fontfamily='serif')   
    fig.text(0.131, 0.89, 'Percent Stacked Bar Chart', fontsize=12, fontfamily='serif')   

    for s in ['top', 'left', 'right', 'bottom']:
        ax.spines[s].set_visible(False)

    fig.text(0.75, 0.9, "Movie", fontweight="bold", fontfamily='serif', fontsize=15, color='#b20710')
    fig.text(0.81, 0.9, "|", fontweight="bold", fontfamily='serif', fontsize=15, color='black')
    fig.text(0.82, 0.9, "TV Show", fontweight="bold", fontfamily='serif', fontsize=15, color='#221f1f')

    fig.text(1.1, 0.93, 'Insight', fontsize=15, fontweight='bold', fontfamily='serif')

    fig.text(1.1, 0.44, '''
    Interestingly, Netflix in India
    is made up nearly entirely of Movies. 

    Bollywood is big business, and perhaps
    the main focus of this industry is Movies
    and not TV Shows.

    South Korean Netflix on the other hand is 
    almost entirely TV Shows.

    The underlying reasons for the difference 
    in content must be due to market research
    conducted by Netflix.
    '''
             , fontsize=12, fontweight='light', fontfamily='serif')

    l1 = lines.Line2D([1, 1], [0, 1], transform=fig.transFigure, figure=fig,color='black',lw=0.2)
    fig.lines.extend([l1])

    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis=u'both', which=u'both',length=0)

    st.pyplot(fig)



def plot_rating_distribution(df):
    order = pd.DataFrame(df.groupby('rating')['count'].sum().sort_values(ascending=False).reset_index())
    rating_order = list(order['rating'])
    mf = df.groupby('type')['rating'].value_counts().unstack().sort_index().fillna(0).astype(int)[rating_order]

    movie = mf.loc['Movie']
    tv = - mf.loc['TV Show']

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.bar(movie.index, movie, width=0.5, color='#b20710', alpha=0.8, label='Movie')
    ax.bar(tv.index, tv, width=0.5, color='#221f1f', alpha=0.8, label='TV Show')

    # Annotations
    for i in tv.index:
        ax.annotate(f"{-tv[i]}", 
                       xy=(i, tv[i] - 60),
                       va='center', ha='center', fontweight='light', fontfamily='serif', color='#4a4a4a')   

    for i in movie.index:
        ax.annotate(f"{movie[i]}", 
                       xy=(i, movie[i] + 60),
                       va='center', ha='center', fontweight='light', fontfamily='serif', color='#4a4a4a')

    for s in ['top', 'left', 'right', 'bottom']:
        ax.spines[s].set_visible(False)

    ax.set_xticklabels(mf.columns, fontfamily='serif')
    ax.set_yticks([])    

    ax.legend().set_visible(False)
    fig.text(0.16, 1, 'Rating distribution by Film & TV Show', fontsize=15, fontweight='bold', fontfamily='serif')
    fig.text(0.16, 0.89, 
    '''We observe that some ratings are only applicable to Movies. 
    The most common for both Movies & TV Shows are TV-MA and TV-14.
    '''
    , fontsize=12, fontweight='light', fontfamily='serif')


    fig.text(0.755, 0.924, "Movie", fontweight="bold", fontfamily='serif', fontsize=15, color='#b20710')
    fig.text(0.815, 0.924, "|", fontweight="bold", fontfamily='serif', fontsize=15, color='black')
    fig.text(0.825, 0.924, "TV Show", fontweight="bold", fontfamily='serif', fontsize=15, color='#221f1f')

    st.pyplot(fig)


def plot_content_over_time(df):
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    color = ["#b20710", "#221f1f"]

    for i, mtv in enumerate(df['type'].value_counts().index):
        mtv_rel = df[df['type']==mtv]['year_added'].value_counts().sort_index()
        ax.plot(mtv_rel.index, mtv_rel, color=color[i], label=mtv)
        ax.fill_between(mtv_rel.index, 0, mtv_rel, color=color[i], alpha=0.9)
    
    ax.yaxis.tick_right()
    
    ax.axhline(y=0, color='black', linewidth=1.3, alpha=.7)

    for s in ['top', 'right', 'bottom', 'left']:
        ax.spines[s].set_visible(False)

    ax.grid(False)

    ax.set_xlim(2008, 2020)
    plt.xticks(np.arange(2008, 2021, 1))

    fig.text(0.13, 0.85, 'Movies & TV Shows added over time', fontsize=15, fontweight='bold', fontfamily='serif')
    fig.text(0.13, 0.59, 
    '''
    We see a slow start for Netflix over several years. 
    Things begin to pick up in 2015 and then there is a 
    rapid increase from 2016.
    Netflix peak global content amount was in 2019.
    It looks like content additions have slowed down in 2020, 
    likely due to the COVID-19 pandemic.
    '''
    , fontsize=12, fontweight='light', fontfamily='serif')

    fig.text(0.13, 0.2, "Movie", fontweight="bold", fontfamily='serif', fontsize=15, color='#b20710')
    fig.text(0.19, 0.2, "|", fontweight="bold", fontfamily='serif', fontsize=15, color='black')
    fig.text(0.2, 0.2, "TV Show", fontweight="bold", fontfamily='serif', fontsize=15, color='#221f1f')

    ax.tick_params(axis=u'both', which=u'both', length=0)

    st.pyplot(fig)



def plot_content_added_polar(data):
    # Check if the DataFrame is empty
    if data.empty:
        st.error("DataFrame is empty. Please provide valid data.")
        return

    # Check if 'type' and 'month_name_added' columns exist in the DataFrame
    if 'type' not in data.columns or 'month_name_added' not in data.columns:
        st.error("DataFrame must contain 'type' and 'month_name_added' columns.")
        return
    
    # Group the data by 'type' and 'month_name_added' and calculate the cumulative sum
    data_sub = data.groupby('type')['month_name_added'].value_counts().unstack().fillna(0).cumsum(axis=0).T

    # Check if 'Movie' and 'TV Show' columns exist in the DataFrame
    if 'Movie' not in data_sub.columns or 'TV Show' not in data_sub.columns:
        st.error("DataFrame must contain columns named 'Movie' and 'TV Show'")
        return
    
    data_sub2 = data_sub.copy()
    data_sub2['Value'] = data_sub2['Movie'] + data_sub2['TV Show']
    data_sub2 = data_sub2.reset_index()
    df_polar = data_sub2.sort_values(by='month_name_added', ascending=False)

    # Define the color map
    color_map = ['#221f1f' for _ in range(12)]
    color_map[0] = color_map[11] = '#b20710'  # highlight December and January in red

    # initialize the figure
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    plt.axis('off')

    # Constants = parameters controlling the plot layout:
    upperLimit = 30
    lowerLimit = 1
    labelPadding = 30

    # Compute max and min in the dataset
    max_value = df_polar['Value'].max()

    # Let's compute heights: they are a conversion of each item value in those new coordinates
    # In our example, 0 in the dataset will be converted to the lowerLimit (10)
    # The maximum will be converted to the upperLimit (100)
    slope = (max_value - lowerLimit) / max_value
    heights = slope * df_polar.Value + lowerLimit

    # Compute the width of each bar. In total we have 2*Pi = 360°
    width = 2 * np.pi / len(df_polar.index)

    # Compute the angle each bar is centered on:
    indexes = list(range(1, len(df_polar.index) + 1))
    angles = [element * width for element in indexes]

    # Draw bars
    bars = ax.bar(
        x=angles,
        height=heights,
        width=width,
        bottom=lowerLimit,
        linewidth=2,
        edgecolor="white",
        color=color_map,
        alpha=0.8
    )

    # Add labels
    for bar, angle, height, label in zip(bars, angles, heights, df_polar["month_name_added"]):
        # Labels are rotated. Rotation must be specified in degrees :(
        rotation = np.rad2deg(angle)

        # Flip some labels upside down
        alignment = "left" if np.pi / 2 <= angle <= 3 * np.pi / 2 else "right"

        # Finally add the labels
        ax.text(
            x=angle,
            y=lowerLimit + bar.get_height() + labelPadding,
            s=label,
            ha=alignment,
            fontsize=10,
            fontfamily='serif',
            va='center',
            rotation=rotation,
            rotation_mode="anchor")

    st.pyplot(plt)









def main():
    st.title("Streamlit Visualization Project")

    # Load the dataset
    data = load_data()

    # Handle missing data and replacements
    data = handle_missing_data(data)

    # Additional feature extraction
    data = add_date_features(data)

    # Display the dataset
    st.write("### Dataset")
    st.write(data)

    # Visualize the ratio of movies and TV shows
    visualize_ratio(data)

    # Quick feature engineering
    data = quick_feature_engineering(data)

    # # Plot top 10 countries
    plot_top_countries(data)

    # # Plot top 10 countries movie & TV show split
    plot_country_movie_tv_ratio(data)

    #plot rating 
    plot_rating_distribution(data)

    #plot timewise uploads
    plot_content_over_time(data)

    #plot month wise
    plot_content_added_polar(data)

    #heatmap
    
    # Add your visualization code here using Streamlit

if __name__ == "__main__":
    main()
