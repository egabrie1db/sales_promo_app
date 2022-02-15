import streamlit as st
import math
from matplotlib.figure import Figure
from tqdm.auto import tqdm
import dateutil
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import ttest_ind
from scipy.stats import mannwhitneyu
import pandas as pd
import numpy as np
import base64

st.set_page_config(layout="wide")


def get_bootstrap(
        data_column_1,
        data_column_2,
        boot_it=1000,
        statistic=np.mean,
        bootstrap_conf_level=0.95
):
    boot_len = max([len(data_column_1), len(data_column_2)])
    boot_data = []
    for i in tqdm(range(boot_it)):
        samples_1 = data_column_1.sample(
            boot_len,
            replace=True
        ).values

        samples_2 = data_column_2.sample(
            boot_len,
            replace=True
        ).values

        boot_data.append(statistic(samples_1-samples_2))
    pd_boot_data = pd.DataFrame(boot_data)

    left_quant = (1 - bootstrap_conf_level)/2
    right_quant = 1 - (1 - bootstrap_conf_level) / 2
    quants = pd_boot_data.quantile([left_quant, right_quant])

    p_1 = norm.cdf(
        x=0,
        loc=np.mean(boot_data),
        scale=np.std(boot_data)
    )
    p_2 = norm.cdf(
        x=0,
        loc=-np.mean(boot_data),
        scale=np.std(boot_data)
    )
    p_value = min(p_1, p_2) * 2

    # Визуализация
    #_, _, bars = plt.hist(pd_boot_data[0], bins = 50)
    fig = Figure()
    #ax = plt.subplots()
    _, _, bars = plt.hist(pd_boot_data[0], bins=50)
    for bar in bars:
        if abs(bar.get_x()) <= quants.iloc[0][0] or abs(bar.get_x()) >= quants.iloc[1][0]:
            bar.set_facecolor('red')
        else:
            bar.set_facecolor('grey')
            bar.set_edgecolor('black')

    plt.style.use('ggplot')
    plt.vlines(quants, ymin=0, ymax=50, linestyle='--')
    plt.xlabel('boot_data')
    plt.ylabel('frequency')
    plt.title("Histogram of boot_data")
    st.pyplot(plt)
    # plt.show()

    st.write({'mean': pd_boot_data[0].mean(),
              # "boot_data": boot_data,
              # "quants": quants,
              "p_value": p_value})
    st.table(quants)


##############################################
df = pd.read_excel('Sales_and_Promotions.xlsx')
###################

# prices = ['Price1', 'Price2', 'Price3',
#         'Price4', 'Price5', 'Price6', 'Price7']
#make_choice_price_sim = st.selectbox('prices of brands', prices, 0)


#st.table(df[df['group_price1'] == 0][make_choice_brand_sim])


col1, col2 = st.beta_columns(2)
with col1:
    #prices = ['Price1', 'Price2', 'Price3', 'Price4', 'Price5', 'Price6', 'Price7']
    brands = ['Brand1', 'Brand2', 'Brand3',
              'Brand4', 'Brand5', 'Brand6', 'Brand7']

    make_choice_price_sim = st.selectbox(
        'choose column of impact', df.columns, 0)
    make_choice_brand_sim = st.selectbox('choose brand', brands, 0)
    df['group_price1'] = np.where(
        df[make_choice_price_sim] < df[make_choice_price_sim].mean(), 0, 1)

    # st.pyplot(plt)
with col2:
    get_bootstrap(
        df[df['group_price1'] == 0][make_choice_brand_sim],
        df[df['group_price1'] == 1][make_choice_brand_sim],
        boot_it=1000,  # amount of bootstrapped datasets
        statistic=np.mean,  # statistics
        bootstrap_conf_level=0.95  # Confidence interval
    )

################################
##############################################
# read campaign starting date
new_date = []
campaign_start_enddate_string = "2014-01-01"
event_date_end = datetime.datetime.strptime(
    campaign_start_enddate_string, "%Y-%m-%d").date()
#event_date_end =campaign_start_enddate_string
weeks = list(range(0, 1))+df['WEEK'].unique().tolist()
weeks = weeks[0:len(weeks)-1]

for i in weeks:
    #i = -i

    i = float(i)
    sc_end_date_minus_11 = pd.to_datetime(
        (event_date_end - dateutil.relativedelta.relativedelta(weeks=-i))).date().strftime("%Y-%m-%d")
    new_date.append(sc_end_date_minus_11)


df['DATE'] = new_date
df['DATE'] = df['DATE'].astype('datetime64')
df['DATE_WEEK_NUMBER'] = df['DATE'].dt.isocalendar().week
df['DATE_year'] = df['DATE'].dt.year
df['DATE_month'] = df['DATE'].dt.month
df['DATUM_week_year'] = df['DATE'].dt.year.astype(int).astype(
    str) + '_' + df['DATE'].dt.isocalendar().week.astype(int).astype(str)  # .zfill(2)
df['DATUM_week_month'] = df['DATE'].dt.year.astype(int).astype(
    str) + '_' + df['DATE'].dt.month.astype(int).astype(str)  # .zfill(2)

# st.table(df)


years = df['DATE_year'].unique()
make_choice_year = st.selectbox('years', years, 0)

brands = ['Brand1', 'Brand2', 'Brand3', 'Brand4', 'Brand5', 'Brand6', 'Brand7',
          'Display1', 'Display2', 'Display3', 'Display4', 'Display5',
          'Display6', 'Display7', 'Price1', 'Price2', 'Price3', 'Price4',
          'Price5', 'Price6', 'Price7', 'Customer'

          ]
make_choice_brand = st.selectbox('brands', brands, 0)


col3, col4, col5 = st.beta_columns(3)
with col3:
    plt.figure(figsize=(12, 8))
    sns.boxplot(x=df['DATE_month'], y=df[make_choice_brand])
    plt.xticks(rotation='vertical')
    st.pyplot(plt)


with col4:
    plt.figure(figsize=(12, 8))
    sns.boxplot(x=df[df['DATE_year'] == make_choice_year]['DATUM_week_month'],
                y=df[df['DATE_year'] == make_choice_year][make_choice_brand])
    plt.xticks(rotation='vertical')
    st.pyplot(plt)


with col5:
    # year
    plt.figure(figsize=(12, 8))
    sns.boxplot(x=df['DATE_year'], y=df[make_choice_brand])
    plt.xticks(rotation='vertical')
    st.pyplot(plt)
