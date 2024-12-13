import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from wordcloud import WordCloud
import os
st.markdown("<h1 style='text-align: center; font-weight: bold;'>🧑‍💻 Exploring Language Trends: A Feature-Based Popularity Analysiss 📊</h1>", unsafe_allow_html=True)
file_path = r'C:\Users\Jhotika Raja\Downloads\dpel\data_cleaned.csv'
try:
    df = pd.read_csv(file_path)
    print("File loaded successfully.")
except FileNotFoundError:
    print("File not found at the specified path:", file_path)
options = st.selectbox("Choose an Option", ["Choose an Option", "📉 Handle Missing Values", "🔍 Explore Dataset", "📊 Exploratory Data Analysis","💡 Top Language Suggestions"])
if options == "📉 Handle Missing Values":
    st.header("🧹 Let's Look at Handling Missing Values")

    # Show the summary of missing values before handling
    st.write("**Missing Values Summary (Before Handling):**")
    st.write(df.isnull().sum())

    # Core categorical columns - filled with mode
    core_categorical_columns = ['Country', 'EdLevel', 'MainBranch', 'NEWCollabToolsHaveWorkedWith']
    for col in core_categorical_columns:
        mode = df[col].mode()[0]
        df[col] = df[col].fillna(mode)

    # Optional categorical columns - filled with 'Not Specified'
    optional_categorical_columns = [
        'CodingActivities', 'DevType', 'LanguageHaveWorkedWith',
        'DatabaseHaveWorkedWith', 'PlatformHaveWorkedWith',
        'WebframeHaveWorkedWith', 'MiscTechHaveWorkedWith',
        'ToolsTechHaveWorkedWith', 'AISearchHaveWorkedWith',
        'NEWCollabToolsWantToWorkWith', 'OpSysProfessional use'
    ]
    for col in optional_categorical_columns:
        df[col] = df[col].fillna('Not Specified')

    # Future preference columns - filled with 'No Preference'
    future_preference_columns = [
        'LanguageWantToWorkWith', 'DatabaseWantToWorkWith',
        'WebframeWantToWorkWith', 'ToolsTechWantToWorkWith',
        'NEWCollabToolsWantToWorkWith'
    ]
    for col in future_preference_columns:
        df[col] = df[col].fillna('No Preference')

    # Convert YearsCode to numeric and fill NaNs with median
    df['YearsCode'] = pd.to_numeric(df['YearsCode'], errors='coerce')
    experience = df['YearsCode'].median()
    df['YearsCode'] = df['YearsCode'].fillna(experience)

    # Drop rows where 'LearnCode' is missing
    df = df.dropna(subset=['LearnCode'])

    # Show updated data and remaining missing values summary
    st.write("**Updated Data After Handling Missing Values:**")
    st.write(df.head())
    st.write("**Remaining Missing Values Summary (After Handling):**")
    st.write(df.isnull().sum())
# Section for "Exploring the dataset"
elif options == "🔍 Explore Dataset":
    st.header("📊 Exploring the Dataset")

    # Add button for displaying value counts and plot for 'YearsCode'
    if st.button("Show 'YearsCode' Value Counts and Plot"):
        st.write(df['YearsCode'].value_counts())

    # Add button for displaying value counts and plot for 'LearnCode' (First Column)
    if st.button("Show 'LearnCode' Value Counts and Plot"):
        learncode_split = df['LearnCode'].str.split(';', expand=True)
        learncode0 = learncode_split[0]
        learncode0_counts = learncode0.value_counts()
        st.write("LearnCode First Column Value Counts:")
        st.write(learncode0_counts)

    # Add button for displaying value counts for 'LanguageHaveWorkedWith'
    if st.button("Show 'LanguageHaveWorkedWith' Value Counts"):
        def count_total_language_have_worked_with(df):
            language_have_total = df['LanguageHaveWorkedWith'].str.split(';', expand=True).stack()
            language_have_counts = language_have_total.value_counts()
            st.write("Total counts in 'LanguageHaveWorkedWith':")
            st.write(language_have_counts)


        count_total_language_have_worked_with(df)

    # Add button for displaying value counts for 'LanguageWantToWorkWith'
    if st.button("Show 'LanguageWantToWorkWith' Value Counts"):
        def count_total_language_want_to_work_with(df):
            language_want_total = df['LanguageWantToWorkWith'].str.split(';', expand=True).stack()
            language_want_counts = language_want_total.value_counts()
            st.write("Total counts in 'LanguageWantToWorkWith':")
            st.write(language_want_counts)


        count_total_language_want_to_work_with(df)

    # Add button for displaying value counts for 'DatabaseHaveWorkedWith'
    if st.button("Show 'DatabaseHaveWorkedWith' Value Counts"):
        def count_total_database_have_worked_with(df):
            database_have_total = df['DatabaseHaveWorkedWith'].str.split(';', expand=True).stack()
            database_have_counts = database_have_total.value_counts()
            st.write("Total counts in 'DatabaseHaveWorkedWith':")
            st.write(database_have_counts)


        count_total_database_have_worked_with(df)

    # Add button for displaying value counts for 'DatabaseWantToWorkWith'
    if st.button("Show 'DatabaseWantToWorkWith' Value Counts"):
        def count_total_database_want_to_work_with(df):
            database_want_total = df['DatabaseWantToWorkWith'].str.split(';', expand=True).stack()
            database_want_counts = database_want_total.value_counts()
            st.write("Total counts in 'DatabaseWantToWorkWith':")
            st.write(database_want_counts)


        count_total_database_want_to_work_with(df)

    # Add button for displaying value counts for 'PlatformHaveWorkedWith'
    if st.button("Show 'PlatformHaveWorkedWith' Value Counts"):
        def count_total_platform_have_worked_with(df):
            platform_have_total = df['PlatformHaveWorkedWith'].str.split(';', expand=True).stack()
            platform_have_counts = platform_have_total.value_counts()
            st.write("Total counts in 'PlatformHaveWorkedWith':")
            st.write(platform_have_counts)


        count_total_platform_have_worked_with(df)

    # Add button for displaying value counts for 'WebframeHaveWorkedWith'
    if st.button("Show 'WebframeHaveWorkedWith' Value Counts"):
        def count_total_webframe_have_worked_with(df):
            webframe_have_total = df['WebframeHaveWorkedWith'].str.split(';', expand=True).stack()
            webframe_have_counts = webframe_have_total.value_counts()
            st.write("Total counts in 'WebframeHaveWorkedWith':")
            st.write(webframe_have_counts)


        count_total_webframe_have_worked_with(df)

    # Add button for displaying value counts for 'WebframeWantToWorkWith'
    if st.button("Show 'WebframeWantToWorkWith' Value Counts"):
        def count_total_webframe_want_to_work_with(df):
            webframe_want_total = df['WebframeWantToWorkWith'].str.split(';', expand=True).stack()
            webframe_want_counts = webframe_want_total.value_counts()
            st.write("Total counts in 'WebframeWantToWorkWith':")
            st.write(webframe_want_counts)


        count_total_webframe_want_to_work_with(df)

    # Add button for displaying value counts for 'DevType'
    if st.button("Show 'DevType' Value Counts"):
        def count_total_dev_type(df):
            dev_type_total = df['DevType'].str.split(';', expand=True).stack()
            dev_type_counts = dev_type_total.value_counts()
            st.write("Total counts in 'DevType':")
            st.write(dev_type_counts)


        count_total_dev_type(df)

    # Add button for displaying value counts for 'CollabToolsHaveWorkedWith'
    if st.button("Show 'CollabToolsHaveWorkedWith' Value Counts"):
        def count_total_collab_tools_have_worked_with(df):
            collab_tools_total = df['NEWCollabToolsHaveWorkedWith'].str.split(';', expand=True).stack()
            collab_tools_counts = collab_tools_total.value_counts()
            st.write("Total counts in 'NEWCollabToolsHaveWorkedWith':")
            st.write(collab_tools_counts)


        count_total_collab_tools_have_worked_with(df)

    # Add button for displaying value counts for 'CollabToolsWantToWorkWith'
    if st.button("Show 'CollabToolsWantToWorkWith' Value Counts"):
        def count_total_collab_tools_want_to_work_with(df):
            collab_tools_want_total = df['NEWCollabToolsWantToWorkWith'].str.split(';', expand=True).stack()
            collab_tools_want_counts = collab_tools_want_total.value_counts()
            st.write("Total counts in 'NEWCollabToolsWantToWorkWith':")
            st.write(collab_tools_want_counts)


        count_total_collab_tools_want_to_work_with(df)

    # Add button for displaying value counts for 'ToolsTechHaveWorkedWith'
    if st.button("Show 'ToolsTechHaveWorkedWith' Value Counts"):
        def count_total_tools_tech_have_worked_with(df):
            tools_tech_have_total = df['ToolsTechHaveWorkedWith'].str.split(';', expand=True).stack()
            tools_tech_have_counts = tools_tech_have_total.value_counts()
            st.write("Total counts in 'ToolsTechHaveWorkedWith':")
            st.write(tools_tech_have_counts)


        count_total_tools_tech_have_worked_with(df)

    # Add button for displaying value counts for 'ToolsTechWantToWorkWith'
    if st.button("Show 'ToolsTechWantToWorkWith' Value Counts"):
        def count_total_tools_tech_want_to_work_with(df):
            tools_tech_want_total = df['ToolsTechWantToWorkWith'].str.split(';', expand=True).stack()
            tools_tech_want_counts = tools_tech_want_total.value_counts()
            st.write("Total counts in 'ToolsTechWantToWorkWith':")
            st.write(tools_tech_want_counts)


        count_total_tools_tech_want_to_work_with(df)

    # Add button for displaying value counts for 'AISearchHaveWorkedWith'
    if st.button("Show 'AISearchHaveWorkedWith' Value Counts"):
        def count_total_ai_search_worked_with(df):
            ai_search_total = df['AISearchHaveWorkedWith'].str.split(';', expand=True).stack()
            ai_search_counts = ai_search_total.value_counts()
            st.write("Total counts in 'AISearchHaveWorkedWith':")
            st.write(ai_search_counts)


        count_total_ai_search_worked_with(df)

    # Add button for displaying value counts for 'MiscTechHaveWorkedWith'
    if st.button("Show 'MiscTechHaveWorkedWith' Value Counts"):
        def count_total_misc_tech_have_worked_with(df):
            misc_tech_have_total = df['MiscTechHaveWorkedWith'].str.split(';', expand=True).stack()
            misc_tech_have_counts = misc_tech_have_total.value_counts()
            st.write("Total counts in 'MiscTechHaveWorkedWith':")
            st.write(misc_tech_have_counts)


        count_total_misc_tech_have_worked_with(df)

    # Add button for displaying value counts for 'OpSysProfessional use'
    if st.button("Show 'OpSysProfessional use' Value Counts"):
        def count_total_op_sys_professional_use(df):
            op_sys_professional_total = df['OpSysProfessional use'].str.split(';', expand=True).stack()
            op_sys_professional_counts = op_sys_professional_total.value_counts()
            st.write(op_sys_professional_counts)
        count_total_op_sys_professional_use(df)
elif options == "📊 Exploratory Data Analysis":
    st.write("📈 This exploratory data analysis (EDA) aims to uncover key trends in developer preferences for programming languages, databases, frameworks, and tools. By examining factors such as experience level, job role, platform usage, and regional differences, the analysis provides insights into the evolving landscape of technology adoption. We explore the popularity of languages like JavaScript and Python and investigate how preferences vary across different developer profiles and geographic regions. Through models assessing cross-platform consistency, we evaluate the versatility of languages across web, mobile, and cloud environments. The analysis also contrasts emerging technologies with legacy tools, offering valuable insights into shifting industry trends. This project provides a comprehensive view of the current and future state of developer tool and language adoption.")
    st.write("Years of code is the only numerical column in the dataset lets explore it!! ")
    if st.button("Distribution curve of Years of Code"):
        df['YearsCode'] = pd.to_numeric(df['YearsCode'], errors='coerce')
        df_cleaned = df.dropna(subset=['YearsCode'])
        plt.figure(figsize=(10, 6))
        sns.histplot(df_cleaned['YearsCode'], kde=True,color='red')
        sns.kdeplot(df_cleaned['YearsCode'], color='yellow')
        plt.title('Distribution of Years of Code')
        plt.xlabel('Years of Code')
        plt.ylabel('Frequency')
        st.pyplot(plt)
        st.markdown("""
        ### Insights from Coding Experience Distribution:

        - **Peak Coding Experience**: The most common range for coding experience is between 5 and 10 years. This range has the highest count, indicating that many respondents are relatively early to mid-career professionals.

        - **Drop in Frequency Beyond 10 Years**: There is a noticeable decline in the number of respondents with more than 10 years of experience. This could indicate that fewer people have longer careers in coding or that some might shift to other roles as they gain experience.
        """)
    if st.button("Boxplot on years of code"):
        df['YearsCode'] = pd.to_numeric(df['YearsCode'], errors='coerce')
        df_cleaned = df.dropna(subset=['YearsCode'])
        plt.figure(figsize=(12, 6))
        sns.boxplot(x=df_cleaned['YearsCode'], color='yellow')
        plt.xlim(0, 50)
        plt.xticks(ticks=range(0, 51, 10))
        plt.title("Boxplot of Years of Code Experience")
        plt.xlabel("Years of Code")
        plt.ylabel("Frequency")
        st.pyplot(plt)
        st.markdown("""
        ### Insights from Coding Experience Boxplot:

        - **Interquartile Range (IQR)**: The yellow box spans from roughly 5 to 20 years, representing the middle 50% of respondents' experience.

        - **Median**: The line inside the box shows a median of around 10 years, indicating a typical experience level of 10 years.

        - **Outliers**: Dots beyond the whiskers represent outliers, with some respondents having over 40 years of experience.

        - **Skewed Distribution**: The longer right tail suggests fewer respondents with very high experience levels, indicating a right-skewed distribution.
        """)
    if st.button("Learning Methods by Years of Coding Experience"):
            df['YearsCode'] = pd.to_numeric(df['YearsCode'], errors='coerce')
            df_cleaned = df.dropna(subset=['YearsCode'])
            df_cleaned['Experience_Bucket'] = pd.cut(
                df_cleaned['YearsCode'],
                bins=[0, 2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, float('inf')],
                labels=['0-2', '3-5', '6-10', '11-15', '16-20', '21-25', '26-30', '31-35', '36-40', '41-45',
                        '46-50', '50+'],
                right=False)
            df_learn = df_cleaned['LearnCode'].str.split(';', expand=True)
            df_learn.columns = [f'LearnCode{i}' for i in range(df_learn.shape[1])]
            learn_methods = df_learn.melt(value_name="LearnMethod").dropna()
            learn_methods_filtered = learn_methods[learn_methods['LearnMethod'] != "Other (please specify):"]
            experience_buckets_repeated = np.tile(df_cleaned['Experience_Bucket'].values,
                                                  len(learn_methods_filtered) // len(df_cleaned))

            # If lengths don't match, adjust to fit the size of 'learn_methods_filtered'
            if len(experience_buckets_repeated) < len(learn_methods_filtered):
                experience_buckets_repeated = np.concatenate([experience_buckets_repeated,
                                                              df_cleaned['Experience_Bucket'].values[
                                                              :len(learn_methods_filtered) - len(
                                                                  experience_buckets_repeated)]])
            learn_methods_filtered['Experience_Bucket'] = experience_buckets_repeated
            learn_method_counts = learn_methods_filtered.groupby(
                ['LearnMethod', 'Experience_Bucket']).size().reset_index(name='Count')
            pivot_data = learn_method_counts.pivot(index='LearnMethod', columns='Experience_Bucket',
                                                   values='Count').fillna(0)

            plt.figure(figsize=(12, 8))
            pivot_data.plot(kind='barh', stacked=True, cmap='viridis')
            plt.xlabel('Frequency')
            plt.ylabel('Learning Method')
            plt.title('Learning Methods by Years of Coding Experience')
            plt.legend(title='Experience Bucket', bbox_to_anchor=(1.05, 1), loc='upper left')
            st.pyplot(plt)
            st.markdown("""
            ### Insights from Coding Learning Methods by Experience Level:

            - **Top Methods**: School, online resources, courses, and on-the-job training are the most popular across all experience levels.

            - **On-the-job Training**: Valued by both new and experienced developers, showing its importance for ongoing skill development.

            - **School & Online Resources**: Especially popular among newer coders, who may rely on structured learning initially.

            - **Hackathons & Bootcamps**: Preferred by those with 0–10 years of experience, providing practical skills for early-career developers.

            - **Books/Physical Media**: Less common but used across all levels, likely favored by those who prefer in-depth learning.

            - **Experienced Coders**: Continue to use diverse methods, highlighting the need for lifelong learning in tech.
            """)
    if st.button("Correlation between Coding Activities and Language Preferences"):

        df_split = df['CodingActivities'].str.split(';', expand=True)
        df_split.columns = [f'Coding_activity{i}' for i in range(df_split.shape[1])]
        index = df.columns.get_loc('CodingActivities') + 1
        for i, col in enumerate(df_split.columns):
            df.insert(index + i, df_split.columns[i], df_split[col])
        original_columns = df.columns.difference(df_split.columns, sort=False)
        selected_columns = ['Coding_activity0', 'Coding_activity1']
        final_columns = original_columns.tolist() + selected_columns
        filtered_df = df[final_columns]
        combined_activities = pd.concat([df['Coding_activity0'], df['Coding_activity1']]).dropna()
        distinct_activities = combined_activities[combined_activities != "Other (please specify):"].unique()
        activity_count = {activity: combined_activities[combined_activities == activity].count() for activity in
                          distinct_activities}
        activity_counts_df = pd.DataFrame(activity_count.items(), columns=['Activity', 'Frequency']).sort_values(
            by='Frequency', ascending=False)
        activities = pd.get_dummies(filtered_df[['Coding_activity0', 'Coding_activity1']], prefix='Activity')
        df_activities = activities.T.groupby(level=0).max().T
        language_want = filtered_df['LanguageWantToWorkWith'].str.get_dummies(sep=';')
        language_have = filtered_df['LanguageHaveWorkedWith'].str.get_dummies(sep=';')
        df_activities = df_activities.reindex(filtered_df.index, fill_value=0)
        language_want = language_want.reindex(filtered_df.index, fill_value=0)
        language_have = language_have.reindex(filtered_df.index, fill_value=0)
        df_with_dummies = pd.concat(
            [filtered_df, df_activities, language_want.add_prefix('want_'), language_have.add_prefix('Have_')],
            axis=1)
        activity_col = df_activities.columns
        language_col = [col for col in df_with_dummies.columns if
                        col.startswith('Have_') or col.startswith('want_')]
        corr_matrix = df_with_dummies[activity_col.union(language_col)].corr().loc[activity_col, language_col]
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0)
        plt.title('Correlation between Language Preference and Coding Activity')
        st.pyplot(plt)
        st.markdown("""
        ### Insights from Professional and Personal Coding Activities:

        - **Professional and Academic Work**: These activities show moderate correlations with several languages. This could suggest that developers working in professional or academic settings often use a diverse range of languages, likely for specific tasks or projects.

        - **Hobby Coding**: Languages commonly associated with hobby projects, like Python and JavaScript, have slight positive correlations with hobby-related coding activities, indicating that these are popular choices for personal or non-commercial projects.

        - **Freelance/Contract Work**: Some languages show positive correlations here, suggesting that developers often choose particular languages when working as freelancers, possibly due to demand or project-specific needs.

        - **Open Source Contributions**: There is a slight positive correlation between languages like Python and JavaScript and contributions to open-source projects. This may indicate these languages are popular for open-source involvement.

        - **Learning and Self-Development**: Languages that are positively correlated with professional development activities (e.g., online courses) might be in high demand for skill-building, with Python and JavaScript often favored by beginners and learners.
        """)
    if st.button("Top Language Popularity Across Experience Levels"):
        filtered_df =df.copy()
        bins = [0, 2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
        labels = ['0-2', '3-5', '6-10', '11-15', '16-20', '21-25', '26-30', '31-35', '36-40', '41-45', '46-50']
        filtered_df['YearsCode'] = pd.to_numeric(filtered_df['YearsCode'], errors='coerce')
        filtered_df = filtered_df.dropna(subset=['YearsCode'])
        filtered_df['Experience_Bucket'] = pd.cut(filtered_df['YearsCode'], bins=bins, labels=labels, right=False)
        language_have = filtered_df['LanguageHaveWorkedWith'].str.get_dummies(sep=';')
        language_popularity_by_experience = pd.concat([filtered_df['Experience_Bucket'], language_have], axis=1)
        language_popularity_by_experience = language_popularity_by_experience.groupby(
            'Experience_Bucket').mean() * 100
        language_popularity_start_end = language_popularity_by_experience.iloc[[0, -1]]
        plt.figure(figsize=(14, 12))
        colors = sns.color_palette("tab20", n_colors=len(language_popularity_start_end.columns))
        for i, (language, color) in enumerate(zip(language_popularity_start_end.columns, colors)):
            plt.plot(
                language_popularity_start_end.index,
                language_popularity_start_end[language],
                marker="o",
                label=language,
                color=color,
                linewidth=2
            )
            plt.text(
                x=language_popularity_start_end.index[-1],
                y=language_popularity_start_end[language].iloc[-1],
                s=language,
                color=color,
                va="center",
                fontsize=9,
            )

        plt.title("Slope Chart of Language Popularity from Low to High Experience Levels")
        plt.xlabel("Experience Level")
        plt.ylabel("Popularity (%)")
        plt.xticks(rotation=45)
        plt.legend().set_visible(False)
        plt.tight_layout()
        st.pyplot(plt)

    if st.button("Language Aspiration from Low to High Experience"):

        # Step 1: Convert 'YearsCode' to numeric, forcing non-numeric values to NaN

        filtered_df = df.copy()
        filtered_df['YearsCode'] = pd.to_numeric(filtered_df['YearsCode'], errors='coerce')
        # Step 2: Drop rows with NaN values in 'YearsCode' (to avoid issues during binning)
        filtered_df = filtered_df.dropna(subset=['YearsCode'])

        # Step 3: Define the bins and labels for Experience Buckets
        bins = [0, 2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
        labels = ['0-2', '3-5', '6-10', '11-15', '16-20', '21-25', '26-30', '31-35', '36-40', '41-45', '46-50']

        # Step 4: Create the 'Experience_Bucket' column
        filtered_df['Experience_Bucket'] = pd.cut(filtered_df['YearsCode'], bins=bins, labels=labels, right=False)

        # Step 5: Process 'LanguageWantToWorkWith' into dummy variables (for Language Aspiration)
        language_want = filtered_df['LanguageWantToWorkWith'].str.get_dummies(sep=';')

        # Step 6: Combine the experience bucket and language data
        language_popularity_want = pd.concat([filtered_df['Experience_Bucket'], language_want], axis=1)

        # Step 7: Group by experience bucket and calculate the mean popularity for each language
        language_popularity_by_experience_want = language_popularity_want.groupby('Experience_Bucket').mean() * 100

        # Step 8: Extract the first and last experience levels for the slope chart
        language_popularity_start_end_want = language_popularity_by_experience_want.iloc[[0, -1]]

        # Step 9: Plot the slope chart for language aspiration across experience levels
        plt.figure(figsize=(18, 12))
        colors = sns.color_palette("tab20", n_colors=len(language_popularity_by_experience_want.columns))

        # Step 10: Plot each language's data points
        for i, (language, color) in enumerate(zip(language_popularity_by_experience_want.columns, colors)):
            plt.plot(
                language_popularity_start_end_want.index,
                language_popularity_start_end_want[language],
                marker="o",
                color=color,
                linewidth=2
            )
            # Add text annotations for each language
            plt.text(
                x=language_popularity_start_end_want.index[-1],
                y=language_popularity_start_end_want[language].iloc[-1],
                s=language,
                color=color,
                va="center",
                fontsize=9,
            )

        # Step 11: Customize the plot's appearance
        plt.title("Slope Chart of Language Aspiration from Low to High Experience Levels", fontsize=16)
        plt.xlabel("Experience Level", fontsize=14)
        plt.ylabel("Popularity (%)", fontsize=14)
        plt.xticks(rotation=45, fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend().set_visible(False)

        # Step 12: Display the plot in Streamlit
        plt.tight_layout()
        st.pyplot(plt)
        st.markdown("""
        ### Insights on Language Popularity Across Experience Levels:

        - **SQL and Python**: Both maintain high popularity across all experience levels, showing steady usage and demand from entry-level to experienced developers.

        - **JavaScript and HTML/CSS**: These languages start with high popularity among beginners but see a slight decline at higher experience levels, suggesting they might be more introductory languages.

        - **Shell (Bash/Unix)**: Gains popularity as experience increases, likely because it is often used in more advanced system operations and automation tasks.

        - **C++ and TypeScript**: These languages are stable but have moderate popularity, appealing to developers with specific technical needs or interests.

        - **Niche or Specialized Languages (e.g., Assembly and Swift)**: Show lower overall popularity, indicating limited use in specialized areas rather than general software development.
        """)
    if st.button("View Top Languages by Experience and Aspiration Across Education Levels"):
        filtered_df=df.copy()
        language_have = filtered_df['LanguageHaveWorkedWith'].str.get_dummies(sep=';')
        language_want = filtered_df['LanguageWantToWorkWith'].str.get_dummies(sep=';')
        top_3_have_languages = language_have.sum().nlargest(3).index.tolist()
        top_3_want_languages = language_want.sum().nlargest(3).index.tolist()
        filtered_df_expanded = filtered_df.copy()
        filtered_df_expanded['EdLevel'] = filtered_df_expanded['EdLevel'].fillna('').str.split(';')
        filtered_df_expanded = filtered_df_expanded.explode('EdLevel')
        filtered_df_expanded = pd.concat([filtered_df_expanded, language_have, language_want], axis=1)
        have_counts = filtered_df_expanded.groupby('EdLevel')[top_3_have_languages].sum()
        want_counts = filtered_df_expanded.groupby('EdLevel')[top_3_want_languages].sum()
        fig, axes = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
        have_counts.plot(kind='bar', stacked=True, ax=axes[0], colormap='Blues')
        axes[0].set_title("Top 3 'Have Worked With' Languages by Educational Level")
        axes[0].set_ylabel("Count of Respondents")
        axes[0].legend(title="Languages")
        want_counts.plot(kind='bar', stacked=True, ax=axes[1], colormap='Oranges')
        axes[1].set_title("Top 3 'Want to Work With' Languages by Educational Level")
        axes[1].set_ylabel("Count of Respondents")
        axes[1].legend(title="Languages")
        plt.xlabel("Educational Level")
        plt.xticks(rotation=90)
        plt.tight_layout()
        st.pyplot(fig)
        st.markdown("""
        ### Overall Observations:

        - **JavaScript Dominance**: It's evident that JavaScript is the most popular language across all educational levels, both in terms of having worked with it and wanting to work with it.

        - **Python's Rise**: Python is the second most popular language for both "Have Worked With" and "Want to Work With," indicating its growing popularity and versatility.

        - **HTML/CSS Importance**: HTML/CSS is a significant language, especially for those with lower educational levels. This suggests its importance for web development and design roles.

        - **TypeScript Gaining Ground**: TypeScript is showing promise, especially among those with higher education levels, indicating its potential as a preferred language for complex web applications.
        """)

        # Insights by Educational Level
        st.markdown("""
        ### Insights by Educational Level:

        - **Primary/Elementary School**:
            - **HTML/CSS**: It is the most popular language for this group, likely due to its accessibility and use in early education.
            - **JavaScript**: It is the second most popular, showing that even at this early stage, there is interest in web development.

        - **Secondary School**:
            - **JavaScript**: It becomes the dominant language, reflecting its widespread use in web development and programming education.
            - **HTML/CSS**: Remains important, likely due to its foundational nature in web development.

        - **Some College/University Study**:
            - **JavaScript**: Continues to be the top choice, solidifying its position as a core language for web development.
            - **Python**: Gains significant traction, likely due to its use in data science, machine learning, and other fields.

        - **Associate Degree**:
            - **JavaScript**: Remains the most popular, indicating its relevance for various technical roles.
            - **Python**: Continues to grow in popularity, potentially due to its use in data analysis and automation.

        - **Bachelor's Degree**:
            - **JavaScript**: Maintains its dominance, reflecting its importance in web development and beyond.
            - **Python**: Shows a strong presence, likely due to its use in data science, machine learning, and other fields.

        - **Master's Degree**:
            - **JavaScript**: Remains the top choice, emphasizing its versatility and applicability in various domains.
            - **TypeScript**: Gains significant popularity, suggesting its preference for large-scale, complex web applications.

        - **Professional Degree (JD, MD, Ph.D, Ed.D, etc.)**:
            - **JavaScript**: Continues to be the most popular, possibly due to its use in research and data analysis.
            - **Python**: Shows a strong presence, likely due to its use in data science and other research-related fields.
        """)
    if st.button("Show Open-Source Contribution Rate by Language Category"):
        df_split = df['CodingActivities'].str.split(';', expand=True)
        df_split.columns = [f'Coding_activity{i}' for i in range(df_split.shape[1])]
        index = df.columns.get_loc('CodingActivities') + 1
        for i, col in enumerate(df_split.columns):
            df.insert(index + i, df_split.columns[i], df_split[col])
        original_columns = df.columns.difference(df_split.columns, sort=False)
        selected_columns = ['Coding_activity0', 'Coding_activity1']
        final_columns = original_columns.tolist() + selected_columns
        filtered_df = df[final_columns]

        # Create the 'ContributesOpenSource' column based on 'Coding_activity0' and 'Coding_activity1'
        filtered_df['ContributesOpenSource'] = (
                (filtered_df['Coding_activity0'] == 'Contribute to open-source projects') |
                (filtered_df['Coding_activity1'] == 'Contribute to open-source projects')
        ).astype(int)
        languages_have = filtered_df['LanguageHaveWorkedWith'].str.get_dummies(sep=';')
        languages_want = filtered_df['LanguageWantToWorkWith'].str.get_dummies(sep=';')
        top_3_have_languages = languages_have.sum().nlargest(3).index.tolist()
        top_3_want_languages = languages_want.sum().nlargest(3).index.tolist()
        filtered_df['UsesTopHaveLanguage'] = languages_have[top_3_have_languages].sum(axis=1) > 0
        filtered_df['UsesTopWantLanguage'] = languages_want[top_3_want_languages].sum(axis=1) > 0

        # Calculate open-source contribution rates for 'Top "Have Worked With"' and 'Top "Want to Work With"'
        have_contribution_rate = filtered_df[filtered_df['UsesTopHaveLanguage']]['ContributesOpenSource'].mean()
        want_contribution_rate = filtered_df[filtered_df['UsesTopWantLanguage']]['ContributesOpenSource'].mean()

        # Create a DataFrame to display contribution rates
        contribution_rates = pd.DataFrame({
            'Language Category': ['Top "Have Worked With"', 'Top "Want to Work With"'],
            'Open Source Contribution Rate': [have_contribution_rate, want_contribution_rate]
        })

        # Plot the bar graph for open-source contribution rate
        plt.figure(figsize=(10, 6))
        plt.bar(contribution_rates['Language Category'], contribution_rates['Open Source Contribution Rate'],
                color=['blue', 'orange'])
        plt.title("Open-Source Contribution Rate by Language Popularity Category")
        plt.ylabel("Average Contribution Rate")
        plt.xlabel("Language Popularity Category")
        plt.ylim(0, 1)
        st.pyplot(plt)
        st.markdown("""
        ### Overall Observations:

        - **Open-Source Contributions**: The plot shows the average open-source contribution rate for languages categorized as "Top 'Have Worked With'" and "Top 'Want to Work With'" categories.

        - **Contribution Rate Similarity**: Interestingly, there is a very close similarity in the average contribution rate between these two categories, indicating that developers who have worked with popular languages are equally likely to contribute to open-source projects as those who want to work with them.
        """)
    if st.button("Show Difference in Current vs. Desired Usage of Web Frameworks"):
        # Process 'WebframeHaveWorkedWith' and 'WebframeWantToWorkWith' columns
        have_worked = df['WebframeHaveWorkedWith'].str.split(';', expand=True).stack().reset_index(level=1,
                                                                                                   drop=True)
        have_worked.name = 'Framework'

        want_to_work = df['WebframeWantToWorkWith'].str.split(';', expand=True).stack().reset_index(level=1,
                                                                                                    drop=True)
        want_to_work.name = 'Framework'

        # Count occurrences of each framework in both "HaveWorked" and "WantToWork" columns
        have_worked_counts = have_worked.value_counts().reset_index()
        have_worked_counts.columns = ['Framework', 'HaveWorkedCount']

        want_to_work_counts = want_to_work.value_counts().reset_index()
        want_to_work_counts.columns = ['Framework', 'WantToWorkCount']

        # Merge the dataframes on 'Framework' and calculate the gap
        framework_comparison = pd.merge(have_worked_counts, want_to_work_counts, on='Framework',
                                        how='outer').fillna(0)
        framework_comparison['Gap'] = framework_comparison['WantToWorkCount'] - framework_comparison[
            'HaveWorkedCount']
        framework_comparison = framework_comparison.sort_values(by='Gap', ascending=False)

        # Create the bar plot using Plotly
        fig = px.bar(
            framework_comparison,
            x='Gap',
            y='Framework',
            color='Gap',
            color_continuous_scale=px.colors.diverging.RdYlGn,
            title="Difference in Current vs. Desired Usage of Web Frameworks",
            labels={'Gap': 'Interest Gap (Want - Have)', 'Framework': 'Web Framework'}
        )

        # Update the layout of the plot
        fig.update_layout(
            xaxis_title="Interest Gap",
            yaxis_title="Framework",
            coloraxis_showscale=False
        )

        # Display the plot in Streamlit
        st.plotly_chart(fig)
        st.markdown("""
        ### Insights by Framework:

        - **Svelte and Deno Gaining Momentum**: Svelte and Deno have the largest positive interest gaps, suggesting significant growth in developer interest and potential adoption.

        - **Solid.js and Next.js Popularity**: Solid.js and Next.js also have substantial positive interest gaps, indicating a growing trend towards these frameworks.

        - **Stable Frameworks**: Frameworks like Remix, Qwik, Blazor, and Phoenix have moderate positive interest gaps, suggesting a stable level of usage and interest.

        - **Smaller Interest Gaps**: Frameworks like Fastify, Lit, Vue.js, Elm, FastAPI, Nuxt.js have smaller positive interest gaps, indicating a more gradual increase in their popularity.

        - **Declining Interest**: Frameworks like jQuery, Node.js, and WordPress have a significant negative interest gap, indicating a decline in their popularity and a shift towards more modern frameworks.

        - **Stable Frameworks**: Frameworks like React, ASP.NET, Express, and Flask have a relatively small negative interest gap, suggesting a stable level of usage and interest.

        - **Angular and AngularJS**: Angular shows a moderate decline in interest, while AngularJS has a more significant decline, indicating a shift towards newer versions and alternative frameworks.
        """)
    if st.button("Show Regional Web Framework Preferences"):
        # Split 'WebframeHaveWorkedWith' and 'WebframeWantToWorkWith' into individual frameworks
        have_worked_data = df['WebframeHaveWorkedWith'].str.split(';', expand=True).stack().reset_index(level=1,
                                                                                                        drop=True)
        have_worked_data.name = 'Framework'

        want_to_work_data = df['WebframeWantToWorkWith'].str.split(';', expand=True).stack().reset_index(level=1,
                                                                                                         drop=True)
        want_to_work_data.name = 'Framework'

        # Combine the data with 'Country' and create two DataFrames
        df_have_worked = df[['Country']].join(have_worked_data)
        df_have_worked['Preference'] = 'Have Worked With'

        df_want_to_work = df[['Country']].join(want_to_work_data)
        df_want_to_work['Preference'] = 'Want To Work With'

        # Concatenate the two DataFrames into one combined DataFrame
        df_combined = pd.concat([df_have_worked, df_want_to_work])

        # Remove rows with NaN values in 'Country' or 'Framework' columns
        df_combined = df_combined.dropna(subset=['Country', 'Framework'])

        # Create the Treemap plot using Plotly
        fig = px.treemap(
            df_combined,
            path=['Country', 'Preference', 'Framework'],
            title="Regional Web Framework Preferences",
            color_discrete_sequence=px.colors.qualitative.Pastel)

        # Update layout for better margin control
        fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))

        # Display the plot in Streamlit
        st.plotly_chart(fig)
        st.markdown("""
        ### Overall Observations:

        - **React Dominance**: React is the most popular framework in terms of both current usage ("Have Worked With") and desired usage ("Want to Work With") across most regions.

        - **Node.js Prevalence**: Node.js is also widely used and desired, particularly in regions like the United States and India.

        - **jQuery Decline**: jQuery shows a significant decline in popularity, with more developers indicating they have worked with it than wanting to work with it in the future.

        - **Regional Variations**: There are some regional variations in framework preferences, with certain frameworks being more popular in specific regions.
        """)
    if st.button("Show Framework Preferences by Developer Role"):
        # Split 'DevType', 'WebframeHaveWorkedWith', and 'WebframeWantToWorkWith' into individual values
        dev_type_split = df['DevType'].str.split(';', expand=True).stack().reset_index(level=1, drop=True)
        dev_type_split.name = 'Role'

        webframe_have_split = df['WebframeHaveWorkedWith'].str.split(';', expand=True).stack().reset_index(level=1,
                                                                                                           drop=True)
        webframe_have_split.name = 'Framework_Have'

        webframe_want_split = df['WebframeWantToWorkWith'].str.split(';', expand=True).stack().reset_index(level=1,
                                                                                                           drop=True)
        webframe_want_split.name = 'Framework_Want'

        # Create expanded DataFrame with 'Role', 'Framework_Have', 'Framework_Want'
        df_expanded = df[['DevType']].join(dev_type_split).join(webframe_have_split).join(webframe_want_split)

        # Count how many developers of each role have worked with or want to work with each framework
        framework_have_counts = df_expanded.groupby(['Role', 'Framework_Have']).size().reset_index(
            name='Count_Have')
        framework_want_counts = df_expanded.groupby(['Role', 'Framework_Want']).size().reset_index(
            name='Count_Want')

        # Merge the counts for frameworks developers have worked with and want to work with
        framework_counts = pd.merge(framework_have_counts, framework_want_counts,
                                    left_on=['Role', 'Framework_Have'], right_on=['Role', 'Framework_Want'],
                                    how='outer').fillna(0)

        # Rename columns for better clarity
        framework_counts.rename(columns={'Framework_Have': 'Framework'}, inplace=True)

        # Melt the data for easy plotting with Plotly
        framework_counts_long = framework_counts.melt(id_vars=['Role', 'Framework'],
                                                      value_vars=['Count_Have', 'Count_Want'],
                                                      var_name='Type', value_name='Count')

        # Create the bar chart using Plotly
        fig = px.bar(
            framework_counts_long,
            x='Role',
            y='Count',
            color='Framework',
            facet_col='Type',
            labels={'Role': 'Developer Role', 'Count': 'Number of Developers', 'Type': 'Usage Type'},
            barmode='group',
            height=600
        )

        # Update layout for the chart
        fig.update_layout(
            xaxis_title="Developer Role",
            yaxis_title="Framework Count",
            legend_title="Web Framework",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        # Display the chart in Streamlit
        st.plotly_chart(fig)
        st.markdown("### Overall Popularity and Role-Based Preferences")
        st.write("""
        - **Dominant Frameworks**: React, Angular, and Node.js are widely adopted across developer roles.
        - **Role-Specific Choices**:
          - **Front-end**: React, Angular, and Vue.js are popular for UI development.
          - **Back-end/Full-stack**: Node.js, Django, and Spring are preferred for server-side tasks.
          - **Mobile**: React Native and Flutter are favored for cross-platform mobile apps.
        """)

        st.markdown("### Trends and Adoption Insights")
        st.write("""
        - **Growth Indicators**: Frameworks with higher "Want to Work With" values show potential for increased adoption.
        - **Stable vs. Declining**: Stable frameworks have balanced usage and interest, while declining frameworks show higher past use.
        """)

        st.markdown("### Factors Influencing Framework Choices")
        st.write("""
        - **Experience and Complexity**: Framework preferences often align with experience level and project demands.
        - **Community Support**: Frameworks with active communities and resources tend to sustain popularity.
        """)
    if st.button("Show Database Usage Preferences Graph"):
        db_have_split = df['DatabaseHaveWorkedWith'].str.split(';', expand=True).stack().reset_index(level=1, drop=True)
        db_have_split.name = 'Database'
        db_want_split = df['DatabaseWantToWorkWith'].str.split(';', expand=True).stack().reset_index(level=1, drop=True)
        db_want_split.name = 'Database'
        df_have = pd.DataFrame({'Database': db_have_split, 'Preference': 'Current Usage'})
        df_want = pd.DataFrame({'Database': db_want_split, 'Preference': 'Desired Usage'})
        db_preferences = pd.concat([df_have, df_want])
        db_counts = db_preferences.groupby(['Database', 'Preference']).size().reset_index(name='Count')
        db_counts_pivot = db_counts.pivot(index='Database', columns='Preference', values='Count').fillna(0)
        db_counts_pivot['Gap'] = db_counts_pivot['Desired Usage'] - db_counts_pivot['Current Usage']
        fig, ax = plt.subplots(figsize=(12, 8))
        db_counts_pivot[['Current Usage', 'Desired Usage']].plot(kind='bar', stacked=False, width=0.7,
                                                                 color=['skyblue', 'salmon'], edgecolor='black', ax=ax)
        plt.title("Current vs. Desired Usage of Databases")
        plt.xlabel("Database")
        plt.ylabel("Number of Developers")
        plt.legend(title="Preference", loc='upper right')
        plt.xticks(rotation=90, ha='right')
        plt.tight_layout()

        # Show the plot in Streamlit
        st.pyplot(fig)
        st.write("MySQL and PostgreSQL: These relational databases continue to be the preferred choice for many developers, with a stable level of usage and interest.MongoDB and Cassandra: These NoSQL databases have a significant positive interest gap, indicating a growing trend towards their adoption for handling unstructured data and scalability.Cloud-Based Databases: Cloud-based databases like Firebase Realtime Database and Cosmos DB are gaining popularity, especially for real-time applications and large-scale data storage.Traditional SQL Databases: Traditional SQL databases like Microsoft SQL Server and Oracle are still widely used, but their growth in desired usage is less pronounced compared to NoSQL databases.")
    if st.button("Show Regional Database Preferences Treemap"):
        # Prepare the split columns for current and desired database usage
        db_have_split = df['DatabaseHaveWorkedWith'].str.split(';', expand=True).stack().reset_index(level=1,
                                                                                                     drop=True)
        db_have_split.name = 'Database'

        db_want_split = df['DatabaseWantToWorkWith'].str.split(';', expand=True).stack().reset_index(level=1,
                                                                                                     drop=True)
        db_want_split.name = 'Database'

        # Directly assign 'Country' without using .repeat()
        df_have = pd.DataFrame({
            'Country': df.loc[db_have_split.index, 'Country'].values,
            'Database': db_have_split,
            'Preference': 'Current Usage'
        })

        df_want = pd.DataFrame({
            'Country': df.loc[db_want_split.index, 'Country'].values,
            'Database': db_want_split,
            'Preference': 'Aspiration'
        })

        # Concatenate both current and desired DataFrames
        db_preferences = pd.concat([df_have, df_want])

        # Calculate counts for each combination of Country, Database, and Preference
        db_counts = db_preferences.groupby(['Country', 'Database', 'Preference']).size().reset_index(name='Count')

        # Generate the treemap using Plotly Express
        fig = px.treemap(
            db_counts,
            path=['Country', 'Preference', 'Database'],
            values='Count',
            color='Preference',
            color_discrete_map={'Current Usage': 'skyblue', 'Aspiration': 'salmon'},
            title="Regional Database Preferences: Current Usage vs. Aspiration"
        )

        fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))

        # Display the treemap in Streamlit
        st.plotly_chart(fig)

        st.write("# Overall Observations")
        st.write("""
           - **MySQL and PostgreSQL Dominance:** MySQL and PostgreSQL are the most popular databases across most regions, both in terms of current usage and aspiration.
           - **NoSQL Databases Gaining Interest:** NoSQL databases like Redis and MongoDB show a significant increase in aspiration compared to current usage, indicating a growing trend towards their adoption.
           - **Regional Variations:** There are some regional variations in database preferences, with certain databases being more popular in specific regions.
           """)

        st.write("# Specific Insights by Region")
        st.write("""
           - **United States:** PostgreSQL and MySQL are the most popular databases, with a strong preference for PostgreSQL in terms of aspiration.
           - **Europe:** MySQL and PostgreSQL are also popular in European countries, with some variations in the preference for other databases like Redis and MongoDB.
           - **India:** MySQL and PostgreSQL are the most popular databases, with a significant interest gap for PostgreSQL, indicating a growing demand for this database.
           - **Other Regions:** In other regions like Asia and South America, MySQL and PostgreSQL are also prominent, indicating their global appeal.
           """)
    if st.button("Show Relationship between Programming Languages and Databases"):
        # Split 'LanguageHaveWorkedWith' and 'LanguageWantToWorkWith' columns into individual languages
        language_have_split = df['LanguageHaveWorkedWith'].str.split(';', expand=True).stack().reset_index(level=1,
                                                                                                           drop=True)
        language_have_split.name = 'Language'
        language_want_split = df['LanguageWantToWorkWith'].str.split(';', expand=True).stack().reset_index(level=1,
                                                                                                           drop=True)
        language_want_split.name = 'Language'

        # Split 'DatabaseHaveWorkedWith' and 'DatabaseWantToWorkWith' columns into individual databases
        db_have_split = df['DatabaseHaveWorkedWith'].str.split(';', expand=True).stack().reset_index(level=1, drop=True)
        db_have_split.name = 'Database'
        db_want_split = df['DatabaseWantToWorkWith'].str.split(';', expand=True).stack().reset_index(level=1, drop=True)
        db_want_split.name = 'Database'

        # Create separate DataFrames for current and desired language preferences with databases
        df_current = pd.DataFrame({'Language': language_have_split}).join(pd.DataFrame({'Database': db_have_split}),
                                                                          how='inner')
        df_current['Preference'] = 'Current Usage'

        df_desired = pd.DataFrame({'Language': language_want_split}).join(pd.DataFrame({'Database': db_want_split}),
                                                                          how='inner')
        df_desired['Preference'] = 'Desired Usage'

        # Combine the data into one DataFrame and compute usage counts
        combined_df = pd.concat([df_current, df_desired], ignore_index=True)
        usage_counts = combined_df.groupby(['Language', 'Database', 'Preference']).size().reset_index(name='Count')

        # Pivot the data for heatmap display
        pivot_data = usage_counts.pivot_table(index='Language', columns=['Database', 'Preference'], values='Count',
                                              fill_value=0)

        # Generate the heatmap plot
        plt.figure(figsize=(14, 10))
        sns.heatmap(pivot_data, cmap='YlGnBu', linewidths=0.5, linecolor='black', cbar_kws={'label': 'Developer Count'})
        plt.title('Relationship between Programming Languages and Databases')
        plt.xlabel('Database and Preference')
        plt.ylabel('Programming Language')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        # Display the plot in Streamlit
        st.pyplot(plt)

        # Insights
        insights = """
        - **Popular Language-Database Combinations**: Some languages are strongly associated with specific databases in both current and desired usage.
        - **Increasing Demand for NoSQL Databases**: Many developers express interest in using NoSQL databases with languages like JavaScript and Python.
        - **Regional Differences**: In some regions, relational databases are more popular with established languages like Java, while new languages like Go show an increased interest in cloud-native databases.
        """
        st.write(insights)
    if st.button("Show Average Years of Coding Experience by Job Role"):
        filtered_df=df.copy()
        filtered_df['YearsCode'] = pd.to_numeric(filtered_df['YearsCode'], errors='coerce')
        filtered_df = filtered_df.dropna(subset=['YearsCode'])
        average_experience = filtered_df.groupby('DevType')['YearsCode'].mean().sort_values()
        plt.figure(figsize=(14, 8))
        average_experience.plot(kind='barh', color='skyblue')
        plt.title("Average Years of Coding Experience by Job Role")
        plt.xlabel("Average Years of Coding")
        plt.ylabel("Job Role")
        plt.xticks(rotation=0)
        plt.tight_layout()
        st.pyplot(plt)
        st.write("""
        **Overall Observations:**
        - **Senior Developers** or **Lead Developers** typically have the most years of coding experience, often reflecting their career advancement.
        - **Junior Developers** and **Interns** have the least amount of experience, typically corresponding to their early stages in their careers.
        - There is a notable difference in experience levels across various job roles, showing how experience correlates with job responsibilities.
    
        **Specific Insights:**
        - **Lead Developers and Senior Developers** usually spend more time on advanced tasks and mentoring, which contributes to their higher years of experience.
        - **Junior Developers** and **Interns** have a shorter time in the industry, which is expected given the nature of their roles and the stage of their careers.
        - **Mid-level Developers** typically have a moderate range of years of experience, showing a transition phase from entry-level to senior roles.
        """)
    if st.button("Show Language Preferences by Job Role (Treemap)"):
        # Step 1: Expand the rows for DevType and LanguageHaveWorkedWith
        expanded_rows = []

        for idx, row in df.iterrows():
            dev_types = row['DevType'].split(';') if pd.notna(row['DevType']) else []
            languages_have = row['LanguageHaveWorkedWith'].split(';') if pd.notna(row['LanguageHaveWorkedWith']) else []

            for dev_type in dev_types:
                for lang_have in languages_have:
                    expanded_rows.append({
                        'JobRole': dev_type.strip(),
                        'Language': lang_have.strip()
                    })

        # Create a new DataFrame from the expanded rows
        have_summary = pd.DataFrame(expanded_rows)

        # Step 2: Aggregate data by counting occurrences of each language for each job role
        have_summary = have_summary.groupby(['JobRole', 'Language']).size().reset_index(name='Count')

        # Step 3: Create the treemap visualization
        fig = px.treemap(have_summary, path=['JobRole', 'Language'], values='Count',
                         title="Language Preferences by Job Role (Treemap)",
                         color='Count', color_continuous_scale='Viridis')
        fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))

        # Display the treemap in Streamlit
        st.plotly_chart(fig)

        # Add insights after the plot
        st.write("""
        **Overall Observations:**
        - **JavaScript Dominance:** JavaScript is the most popular language across various job roles, particularly for front-end and full-stack developers.
        - **Python's Rise:** Python is also a widely used language, especially for data science, machine learning, and back-end development.
        - **SQL's Importance:** SQL remains a crucial language for database interactions, used across different job roles.
        - **C# and Java Popularity:** C# and Java are popular choices for enterprise development and backend systems.
        - **Language Diversity:** There's a diverse range of languages used across different job roles, reflecting the multifaceted nature of software development.

        **Specific Insights by Job Role:**
        - **Developer, Full-Stack:** JavaScript and SQL are the most popular languages, followed by C#.
        - **Developer, Front-End:** JavaScript and HTML/CSS are the dominant languages, with TypeScript gaining popularity.
        - **Developer, Back-End:** SQL, JavaScript, and Python are widely used, reflecting the diverse nature of back-end development.
        - **Student:** JavaScript, Python, and C++ are popular choices for students, indicating a focus on modern programming languages.
        - **Other Roles:** Other roles like DevOps Specialist, Engineering Manager, and Academic Researcher have varied language preferences, depending on their specific responsibilities.
        """)
    if st.button("Show Heatmap of Trending Programming Language Usage Across Platforms"):
        filtered_df=df.copy()
        platforms_split = filtered_df['PlatformHaveWorkedWith'].str.split(';', expand=True).stack().reset_index(level=1,
                                                                                                                drop=True)
        platforms_split.name = 'Platform'
        languages_split = filtered_df['LanguageHaveWorkedWith'].str.split(';', expand=True).stack().reset_index(level=1,
                                                                                                                drop=True)
        languages_split.name = 'Language'

        # Step 2: Join the expanded data into the DataFrame
        df_expanded = filtered_df[['PlatformHaveWorkedWith', 'LanguageHaveWorkedWith']].join(platforms_split).join(
            languages_split)

        # Step 3: Group by Language and Platform, then create the heatmap data
        heatmap_data = df_expanded.groupby(['Language', 'Platform']).size().unstack(fill_value=0)

        # Step 4: Plot the heatmap
        plt.figure(figsize=(14, 10))
        sns.heatmap(heatmap_data, cmap="YlGnBu", annot=False, fmt="d", cbar_kws={'label': 'Usage Count'})
        plt.title("Heatmap of Trending Programming Language Usage Across Platforms")
        plt.xlabel("Platform")
        plt.ylabel("Language")

        # Display the heatmap in Streamlit
        st.pyplot(plt)

        # Add insights after the plot
        st.write("""
        **Overall Observations:**
        - **JavaScript and Python Dominance:** JavaScript and Python are the most widely used languages across various platforms, indicating their versatility and popularity in web development, data science, and machine learning.
        - **Platform Preferences:** Certain platforms have preferences for specific languages. For example, AWS and Google Cloud are popular with Python developers, while Microsoft Azure is often used with C#.
        - **Language-Platform Combinations:** The heatmap highlights specific language-platform combinations, such as Python with AWS and JavaScript with Vercel.
        - **Less Popular Languages:** Some languages, like Ada, Cobol, and Fortran, have limited usage across platforms, reflecting their niche applications.

        **Specific Insights by Language:**
        - **JavaScript:** JavaScript is widely used across various platforms, particularly for web development and front-end frameworks like React and Angular.
        - **Python:** Python is popular for data science, machine learning, and back-end development, with strong usage on platforms like AWS and Google Cloud.
        - **C#:** C# is commonly used with Microsoft Azure and is popular for enterprise development and game development.
        - **Java:** Java is widely used for enterprise applications and is often associated with platforms like AWS and Microsoft Azure.
        """)
    if st.button("Show Heatmap of Desired Programming Language Usage Across Platforms"):
        filtered_df=df.copy()
        platforms_split = filtered_df['PlatformHaveWorkedWith'].str.split(';', expand=True).stack().reset_index(level=1,
                                                                                                                drop=True)
        platforms_split.name = 'Platform'
        languages_want_split = filtered_df['LanguageWantToWorkWith'].str.split(';', expand=True).stack().reset_index(
            level=1, drop=True)
        languages_want_split.name = 'Language'

        # Step 2: Join the expanded data into the DataFrame
        df_expanded = filtered_df[['PlatformHaveWorkedWith', 'LanguageWantToWorkWith']].join(platforms_split).join(
            languages_want_split)

        # Step 3: Group by Language and Platform, then create the heatmap data
        heatmap_data = df_expanded.groupby(['Language', 'Platform']).size().unstack(fill_value=0)

        # Step 4: Plot the heatmap using a custom color map
        plt.figure(figsize=(14, 10))
        cmap = sns.diverging_palette(220, 10, as_cmap=True)  # Diverging color palette for the heatmap
        sns.heatmap(heatmap_data, cmap=cmap, annot=False, fmt="d", cbar_kws={'label': 'Usage Count'})
        plt.title("Heatmap of Programming Language Usage Across Platforms")
        plt.xlabel("Platform")
        plt.ylabel("Language")

        # Display the heatmap in Streamlit
        st.pyplot(plt)

        # Add insights after the plot
        st.write("""
        **Overall Observations:**
        - **JavaScript and Python Dominance:** JavaScript and Python are the most widely used languages across various platforms, indicating their versatility and popularity in web development, data science, and machine learning.
        - **Platform Preferences:** Certain platforms have preferences for specific languages. For example, AWS and Google Cloud are popular with Python developers, while Microsoft Azure is often used with C#.
        - **Language-Platform Combinations:** The heatmap highlights specific language-platform combinations, such as Python with AWS and JavaScript with Vercel.
        - **Less Popular Languages:** Some languages, like Ada, Cobol, and Fortran, have limited usage across platforms, reflecting their niche applications.

        **Specific Insights by Language:**
        - **JavaScript:** JavaScript is widely used across various platforms, particularly for web development and front-end frameworks like React and Angular.
        - **Python:** Python is popular for data science, machine learning, and back-end development, with strong usage on platforms like AWS and Google Cloud.
        - **C#:** C# is commonly used with Microsoft Azure and is popular for enterprise development and game development.
        - **Java:** Java is widely used for enterprise applications and is often associated with platforms like AWS and Microsoft Azure.
        - **Other Languages:** Other languages like Ruby, Go, and Rust have specific niches and are used on various platforms depending on their strengths and weaknesses.
        """)
    if st.button("Show Popular Collaboration Tools by Developer Type"):
        Colab_Tools = df.groupby(['DevType', 'NEWCollabToolsHaveWorkedWith']).size().reset_index(name='count')
        filtered_Colab_Tools_counts = Colab_Tools.loc[Colab_Tools.groupby('DevType')['count'].idxmax()]
        plt.figure(figsize=(20, 6))
        sns.barplot(x='DevType', y='count', hue='NEWCollabToolsHaveWorkedWith', data=filtered_Colab_Tools_counts)
        plt.title('Programming Language Communities and Collaboration Tools')
        plt.xlabel('Developer Type (DevType)')
        plt.ylabel('Number of Users')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Collaboration Tool Used')
        plt.tight_layout()

        # Display the plot in Streamlit
        st.pyplot(plt)

        # Display insights after the plot
        st.write("""
        **Insights:**

        - **Visual Studio Code is the most popular tool:** Its usage spans across various developer types, reflecting its versatility and widespread adoption.

        - **Android Studio's Specific Audience:** Android Studio is predominantly used by developers focusing on mobile development, aligning with its primary use case.

        - **Xcode's Targeted Use:** Xcode's usage is limited to a smaller group of developers, indicating its specialized use for iOS and macOS development.
        """)
    if st.button("Show Most Popular Programming Language by Country"):
        df1 = df.copy()
        df1['LanguageHaveWorkedWith'] = df['LanguageHaveWorkedWith'].str.split(';').str[0]
        language_counts = df1.groupby(['Country', 'LanguageHaveWorkedWith']).size().reset_index(name='count')
        most_used_language = language_counts.loc[language_counts.groupby('Country')['count'].idxmax()]
        filtered_most_used_Language = most_used_language[most_used_language['count'] > 100]
        plt.figure(figsize=(20, 6))
        sns.barplot(x='Country', y='count', hue='LanguageHaveWorkedWith', data=filtered_most_used_Language)
        plt.title('Most Popular Programming Language by Country')
        plt.xlabel('Country')
        plt.ylabel('Number of Users')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Programming Language')
        st.pyplot(plt.gcf())
        st.markdown("### Insights")
        st.write("""
            - **Wide Variation**: There is a significant variation in the popularity of programming languages across different countries.
            - **HTML/CSS and Bash/Shell**: These two languages are consistently popular across many countries, likely due to their wide applicability in web development and system administration.
            - **C++**: C++ is also popular in several countries, indicating its use in system programming and performance-critical applications.
            """)
    if st.button("Ai frameworks by country"):
        frameworks_data = df['AISearchHaveWorkedWith'].str.split(';', expand=True).stack().reset_index(level=1,
                                                                                                       drop=True)
        frameworks_data.name = 'Framework'
        df_expanded = df[['Country']].join(frameworks_data)
        framework_counts = df_expanded.groupby(['Country', 'Framework']).size().reset_index(name='Count')
        fig = px.bar(
            framework_counts,
            x='Country',
            y='Count',
            color='Framework',
            title='AI Frameworks Usage by Country',
            labels={'Count': 'Usage Count', 'Country': 'Country', 'Framework': 'AI Framework'},
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
        fig.update_layout(
            xaxis_title="Country",
            yaxis_title="Usage Count",
            legend_title="AI Framework",
            barmode='stack',  #
            xaxis={'categoryorder': 'total descending'}
        )
        fig.update_yaxes(range=[0, 2000])
        st.plotly_chart(fig)
        st.markdown("### AI Framework Insights")
        st.write("""
               - **Wide Range of Usage**: The plot shows a wide range of usage for various AI frameworks across different countries.
               - **ChatGPT Dominance**: ChatGPT is the most popular AI framework, with high usage in many countries.
               - **Regional Variations**: There are regional variations in framework usage, with some frameworks being more popular in certain regions.

               #### Specific Insights
               - **ChatGPT**: ChatGPT is the most widely used framework, with high usage in countries like the United States, Canada, and Australia.
               - **Google Bard AI**: Google Bard AI has significant usage in countries like the United States and Canada.
               """)
elif options == "💡 Top Language Suggestions":
    processed_df = df[['DevType', 'LanguageHaveWorkedWith', 'LanguageWantToWorkWith']].copy()
    expanded_rows = []

    # Expand rows for DevType, LanguageHaveWorkedWith, and LanguageWantToWorkWith
    for idx, row in processed_df.iterrows():
        # Add checks for missing values before calling .split(';')
        dev_types = row['DevType'].split(';') if pd.notna(row['DevType']) else []
        languages_have = row['LanguageHaveWorkedWith'].split(';') if pd.notna(row['LanguageHaveWorkedWith']) else []
        languages_want = row['LanguageWantToWorkWith'].split(';') if pd.notna(row['LanguageWantToWorkWith']) else []

        for dev_type in dev_types:
            for lang_have in languages_have:
                expanded_rows.append({
                    'DevType': dev_type.strip(),
                    'Language': lang_have.strip(),
                    'HaveWorkedWith': 1,
                    'WantToWorkWith': 0
                })
            for lang_want in languages_want:
                # Check if language is already added
                row_exists = False
                for r in expanded_rows:
                    if r['DevType'] == dev_type.strip() and r['Language'] == lang_want.strip():
                        r['WantToWorkWith'] = 1
                        row_exists = True
                        break
                if not row_exists:
                    expanded_rows.append({
                        'DevType': dev_type.strip(),
                        'Language': lang_want.strip(),
                        'HaveWorkedWith': 0,
                        'WantToWorkWith': 1
                    })

    # Create a DataFrame from the expanded rows
    combined_df = pd.DataFrame(expanded_rows)

    # Calculate language popularity
    popularity = combined_df.groupby('Language').sum()[['HaveWorkedWith', 'WantToWorkWith']]
    popularity['TotalPopularity'] = popularity['HaveWorkedWith'] + popularity['WantToWorkWith'] * 2
    popular_languages = popularity['TotalPopularity'].sort_values(ascending=False)

    # Assign weights based on popularity
    combined_df = pd.merge(combined_df, popularity[['TotalPopularity']], on='Language', how='left')
    combined_df['Weight'] = combined_df['HaveWorkedWith'] + 2 * combined_df['WantToWorkWith'] + 0.5 * combined_df[
        'TotalPopularity']

    # Create the interaction matrix
    developer_language_matrix = combined_df.pivot_table(index='DevType', columns='Language', values='Weight',
                                                        fill_value=0)

    # Apply SVD with optimal component tuning
    svd = TruncatedSVD(n_components=15)
    latent_matrix = svd.fit_transform(developer_language_matrix)
    latent_languages = svd.components_.T

    # Cosine similarity in the latent space
    similarity_matrix = cosine_similarity(latent_matrix, latent_languages)

    # Generate recommendations for a given developer type
    dev_type_idx = 0  # Example: first developer type
    recommendations = pd.Series(similarity_matrix[dev_type_idx], index=developer_language_matrix.columns).sort_values(
        ascending=False)

    # Display recommendations with insights
    st.markdown("### Top Recommended Languages")
    st.write("Top recommended languages for developers based on popularity and flexibility across different roles:")

    # Display insights
    st.write("""
    - **Top Recommended Languages**: C#, C, and HTML/CSS are highly recommended, aligning well with both developer experience and aspirations.
    - **Most Popular Languages by Total Popularity**: Bash/Shell, C#, and HTML/CSS are widely used across industries, with JavaScript crucial for web development.
    - **Emerging vs. Legacy Languages**: Python ranks lower, showing niche usage beyond data roles, while Go is popular for backend development.
    - **Popularity-Driven Recommendations**: Weighting emphasizes languages like C#, C, and HTML/CSS, which are adaptable across roles.
    - **Word Cloud Visualization**: Larger words indicate higher recommendation scores, highlighting popular languages at a glance.
    """)

    st.write("Top Recommended Languages based on Popularity:", recommendations.head(10))
    st.write("\nLanguage Popularity Scores (for reference):", popular_languages.head(10))
    lang_recommendations = dict(zip(recommendations.index, recommendations.values))
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(
        lang_recommendations)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title("Top Recommended Languages Word Cloud")
    st.pyplot(plt)
