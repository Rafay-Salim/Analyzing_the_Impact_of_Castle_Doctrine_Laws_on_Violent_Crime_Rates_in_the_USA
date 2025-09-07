#!/usr/bin/env python
# coding: utf-8

# # **Analyzing the Impact of Castle Doctrine Laws on Violent Crime Rates in the USA**
# 
# ![fbi.jpg](fbi.jpg)
# ## **Introduction**
# 
# The **Castle Doctrine** is a legal principle that grants individuals the right to use reasonable force, including deadly force, to defend themselves against an intruder within their own homes. Rooted in the notion that one's home is their "castle," these laws eliminate the duty to retreat before using force in self-defense. Proponents argue that the Castle Doctrine empowers lawful homeowners to protect themselves and deters criminal activity, while critics express concerns that such laws may escalate violence and lead to an increase in homicides.
# 
# Understanding the **causal relationship** between the implementation of Castle Doctrine laws and changes in violent crime rates is crucial for policymakers, law enforcement agencies, and communities. By analyzing this relationship, we can assess whether these laws effectively reduce crime or inadvertently contribute to higher rates of violence.
# 
# ## **Dataset Overview**
# 
# This analysis utilizes a comprehensive dataset from the **FBI**, encompassing various states over multiple years. The dataset captures a wide range of variables related to violent crimes, socioeconomic conditions, and demographic factors. Notably, the implementation of Castle Doctrine laws occurred at different times across states, with the majority adopting these statutes around **2006**. This staggered adoption provides a unique opportunity to employ robust methodologies to isolate the effect of these laws on violent crime rates.
# 
# ### **Selected Columns for Analysis**
# 
# For a focused and meaningful analysis, we restrict our examination to the following columns from the dataset:
# 
# | **Column**         | **Description**                                                                                                                                                                                                                 |
# |--------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
# | `year`             | The calendar year of the observation (e.g., 2005, 2010).                                                                                                                                                                       |
# | `post`             | A binary indicator where `1` signifies the post-treatment period (after the implementation of Castle Doctrine laws) and `0` denotes the pre-treatment period.                                                                    |
# | `sid`              | The state identifier, uniquely representing each state in the dataset.                                                                                                                                                       |
# | `homicide`         | The number of homicides recorded in the state per 100,000 population for the given year.                                                                                                                                       |
# | `robbery`          | The number of robberies reported in the state per 100,000 population for the given year.                                                                                                                                        |
# | `larceny`          | The number of larcenies recorded in the state per 100,000 population for the given year.                                                                                                                                        |
# | `assault`          | The number of aggravated assaults reported in the state per 100,000 population for the given year.                                                                                                                              |
# | `burglary`         | The number of burglaries recorded in the state per 100,000 population for the given year.
# | `l_exp_pubwelfare`         | Logged public welfare spending                                                                                                                                        |
# | `l_police`         | Logged police presence                                                                                                                                        |
# | `l_income`         | Logged income                                                                                                                                        |
# 
# | `murder`           | The number of murders reported in the state per 100,000 population for the given year.                                                                                                                                          |
# | `unemployrt`       | The unemployment rate in the state for the given year, serving as an economic indicator.                                                                                                                                          |
# | `poverty`          | The poverty rate in the state for the given year, reflecting socioeconomic conditions.                                                                                                                                           |
# | `blackm_15_24`     | The percentage of Black males aged 15-24 in the state for the given year.                                                                                                                                                        |
# | `whitem_15_24`     | The percentage of White males aged 15-24 in the state for the given year.                                                                                                                                                        |
# | `popwt`     | Population weight                                                                                                                                                                                                                       |
# 
# ### **Crime Definitions**
# 
# To ensure clarity in our analysis, it's essential to define each of the key crime-related variables included in our dataset:
# 
# | **Variable**   | **Definition**                                                                                                                                                                                                                     |
# |----------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
# | **Homicide**   | **Homicide** is defined as the sum of **murder** and **non-negligent manslaughter**. It represents the total number of intentional killings within a state, normalized per 100,000 state population.                                 |
# | **Murder**     | **Murder** refers to the unlawful killing of another human being without justification or valid excuse, committed with the necessary intention as defined by the law in a specific jurisdiction.                                           |
# | **Larceny**    | **Larceny** is the unlawful taking and carrying away of personal property with the intent to deprive the rightful owner of it permanently. It encompasses various forms of theft that do not involve force or intimidation.               |
# | **Assault**    | **Assault** involves the act of causing physical harm or unwanted physical contact to another person. This includes aggravated assaults, which are more severe and may involve the use of weapons or intent to cause serious injury.       |
# | **Burglary**   | **Burglary** is the act of illegally entering a building or other areas without permission, typically with the intention of committing a further criminal offense inside. It does not necessarily involve theft or violence.            |
# | **Robbery**    | **Robbery** is the act of taking property or money from a person through force, intimidation, or threat of violence. Unlike larceny, robbery involves direct confrontation and coercion against the victim.                                |
# 
# 
# ### **Temporal and Spatial Dimensions**
# 
# The dataset spans multiple states across the United States and covers several years, allowing for a longitudinal analysis of crime trends in relation to the implementation of Castle Doctrine laws. Most states adopted these statutes around **2006**, but the exact year of implementation varies, providing a natural experiment setting to evaluate the laws' impact. By comparing states before and after the adoption period and against states that did not adopt the law during the study period, we can effectively employ different techniques to infer causality.
# 
# ---
# 
# 

# In[ ]:


#import as needed
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf


# ## Dataset Exploration

# In[ ]:


# Import dataset
df = pd.read_csv("castle_doctrine_fbi.csv")


# In[ ]:


selected_columns = ['post', 'homicide', 'robbery', 'larceny',
                    'assault', 'burglary', 'murder', 'unemployrt','l_exp_pubwelfare' ,'l_police','l_income',
                    'poverty', 'blackm_15_24', 'whitem_15_24']

#Plot correlation heatmap
df_selected = df[selected_columns]
corr = df_selected.corr()
plt.figure(figsize=(20, 12))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap of Selected Columns')
plt.show()


# 1. Corelation No.1: "poverty" vs "unemployrt"
#     - This is a pretty intuitive and unsuprising corelation as those are unemployed would tend to have lower financial status.
#     - The two are likely directly causally related as unemployment does often directly lead to poverty
#     - There could however be a confounding variable like "l_exp_pubwelfare" which describes the spending on public welfare and that could also have an effect on the poverty status of an individual as well.
# 2. Correlation No.2: "l_exp_pubwelfare" vs "poverty"
#     - There is a moderate corelation l_exp_pubwelfare and poverty. This can be interpretated as such because higher public welfare expenditures might be associated with areas of high poverty because public welfare programs target such regions.
#     - This is a indirect casual analysis as confounder variables like other policy decisions and economic conditions might also play a role however we should still note that the level of poverty still likely drives welfare expenditure.
# 3. Corelation No.3: "l_police" vs "homicide":
#     - There is present, a slight positive contribution between l_police and homicide.
#     - It is very likely that this is not implying that higher police presence is causing higher homicides in such areas but instead due to higher police expenditure towards regions with higher homicide rates, this makes them allocate more resources to law enforcement and having a higher logged presence specifically in that region.
#     - This is likely a an indirect relationship, higher homicide rates result in higher police presence and not the opposite. Many any other confounder variables like poverty rates, unemployment rates and just general economic factors such as the public welfare spent can all high a more significant effect on the homicide rates.
# 4. Corelation No.4: 'poverty' vs 'burglary':
#     - There is quite a high positive corellation between our robbbery and poverty stats.
#     - Interpreating this corellation, one could come to a possible conclusion that lower poverty rates does result in people in great financial despair thus being unable to fullfil their financial needs. Such people resorting resorting to burglary would then make perfect realistic sense.
#     - This is a causal relationship however it may be indirect as confounding variables like the police presence in the area or the unemployment of the person (which would affect their poverty level)
# 5. Corelation No.5: 'poverty' vs 'l_income'
#     - This is a very obvious completely negative corelation.
#     - It doesn't deserve much explanation as any person who logs their income as higher will obviously have a lower poverty level thus this inverse relationship is shown clearly.
#     - While the relationship may be obvious/redundant, I still felt that it was important to show just get an idea of the spectrum and the procedure in which we analyze such heatmaps.
#     - Finally, this is almost definetely direct causal relationship and there should be no confounding variables unless they are forcefully introduced for some other purposes.

# ## **Analyzing the Impact of Castle Doctrine Laws on Homicide Rates**
# 
# We set out to analyze the impact of Castle Doctrine laws on homicide rates across different states in the United States. This investigation is particularly intriguing as it delves into the delicate balance between individual self-defense rights and broader public safety concerns. By examining whether the implementation of these laws correlates with changes in homicide rates, we aim to uncover insights that could inform policymakers and the community. However, we remain cautious, fully aware of the complexities involved in isolating the law's effect amidst a myriad of confounding factors. Despite these challenges, our analysis strives to navigate these intricacies to shed light on the implications of Castle Doctrine legislation.

# In[ ]:


bef_avg_hom = df[df['post'] == 0 ]["homicide"].mean()
aft_avg_hom = df[df['post'] == 1]["homicide"].mean()

ATE = aft_avg_hom - bef_avg_hom
print(ATE)


# 
# Calculate the p-value for the treatment and store it in `p_value`.
# 
# Comment on the statistical significance of your result. What does this p-value say about homicide rates and the impact of the castle doctrine based on our assumptions? Clearly state your null and alternative hypotheses. Should you reject the null hypothesis?
# 

# In[ ]:


import pandas as pd
from scipy.stats import ttest_ind


# Filter and sort by year to get the latest 50 instances for both groups
pre_doctrine = df[df['post'] == 0].sort_values(by='year', ascending=False).head(50)['homicide']
post_doctrine = df[df['post'] == 1].sort_values(by='year', ascending=False).head(50)['homicide']

# Perform independent t-test assuming unequal variances
t_stat, p_value = ttest_ind(pre_doctrine,post_doctrine, equal_var=False)
p_value

# b) As the p-value is less than 0.05, we reject the null hypothesis, indicating that there is a statistically significant difference in homicide rates before and after the doctrine was implemented.


# Calculate the ATE across every state separately.
# 
# Report the CATE. Store it in the `CATE` variable.
# 
# Evidence for the **Simpson's Paradox** for any individual state?

# In[ ]:


states_with_doctrine = df[df['post'] == 1]['sid'].unique()
df_filtered = df[df['sid'].isin(states_with_doctrine)]

sids = []
pre_treatment_avg_homicide = []
post_treatment_avg_homicide = []
ate_homicide = []

for sid, group in df_filtered.groupby('sid'):
    pre_avg = group[group['post'] == 0]['homicide'].mean()
    post_avg = group[group['post'] == 1]['homicide'].mean()
    ate = post_avg - pre_avg

    sids.append(sid)
    pre_treatment_avg_homicide.append(pre_avg)
    post_treatment_avg_homicide.append(post_avg)
    ate_homicide.append(ate)

DF_CATE_STATE = pd.DataFrame({
    'sid': sids,
    'Pre_Treatment_Avg_Homicide': pre_treatment_avg_homicide,
    'Post_Treatment_Avg_Homicide': post_treatment_avg_homicide,
    'ATE_Homicide': ate_homicide
})

DF_CATE_STATE = DF_CATE_STATE.sort_values(by='sid')

CATE = DF_CATE_STATE['ATE_Homicide'].mean()

print(DF_CATE_STATE)
print("CATE:", CATE)

DF_CATE_STATE
CATE


# 1. State-Level ATEs: In the dataframe, most ATE_Homicide values are negative, suggesting a decrease in the homicide rate after treatment. For example:- State 2 has an ATE_Homicide of -0.962524.
#             - State 11 has an ATE_Homicide of -0.631924.
# 2. Combined ATE: The overall CATE (mean of ATE_Homicide across all states) is -0.20842, showing a general reduction in the homicide rate after treatment.
# 
# if you observe that some states with high Pre_Treatment_Avg_Homicide and low treatment effect (ATE_Homicide) disproportionately influence the overall mean, this could lead to Simpson's Paradox such as how State 19 has a high Pre_Treatment_Avg_Homicide (12.569534) but a very low ATE_Homicide (0.013525), which might distort the overall effect when aggregated.
# 
# This could occur due to a multitude of reasons:
# 1. Confounding Factors: Differences in the population or crime prevention methods between states might bias the results.
# 2. Weighting of Aggregates: States with larger Pre_Treatment_Avg_Homicide might have more influence on the overall trend, even if their ATE_Homicide is small.
# 3. Heterogeneity: The treatment may not be equally effective across all states due to local conditions or policies.
# 
# 

# Find the ATE for those states in 2010 (i.e. post treatment group) and 2005 (i.e. pre treatment group),`
# 
# Report the CATE. Store it in the variable `CATE_Y`
# 

# In[ ]:


# Filter the dataset to include only the states that implemented the doctrine in 2010
states_with_doctrine_2010 = df[(df['post'] == 1) & (df['year'] == 2010)]['sid'].unique()
df_filtered = df[df['sid'].isin(states_with_doctrine_2010)]

# Restrict the analysis to the year 2005 for pre-implementation and 2010 for post-implementation
df_pre_2005 = df_filtered[df_filtered['year'] == 2005]
df_post_2010 = df_filtered[df_filtered['year'] == 2010]

# Initialize lists to store the results
sids = []
pre_treatment_avg_homicide = []
post_treatment_avg_homicide = []
ate_homicide = []

# Group the data by state and calculate the required values
for sid in states_with_doctrine_2010:
    pre_avg = df_pre_2005[df_pre_2005['sid'] == sid]['homicide'].mean()
    post_avg = df_post_2010[df_post_2010['sid'] == sid]['homicide'].mean()
    ate = post_avg - pre_avg

    sids.append(sid)
    pre_treatment_avg_homicide.append(pre_avg)
    post_treatment_avg_homicide.append(post_avg)
    ate_homicide.append(ate)

# Create the DataFrame
DF_CATE_STATE_YEAR = pd.DataFrame({
    'sid': sids,
    'Pre_Treatment_Avg_Homicide': pre_treatment_avg_homicide,
    'Post_Treatment_Avg_Homicide': post_treatment_avg_homicide,
    'ATE_Homicide': ate_homicide
})

# Sort the DataFrame by 'sid'
DF_CATE_STATE_YEAR = DF_CATE_STATE_YEAR.sort_values(by='sid')

# Calculate the Conditional Average Treatment Effect (CATE)
CATE_Y = DF_CATE_STATE_YEAR['ATE_Homicide'].mean()

# Print the DataFrame and CATE
print(DF_CATE_STATE_YEAR)
print("CATE_Y:", CATE_Y)

# Store the results in the required variables
DF_CATE_STATE_YEAR
CATE_Y


# The analysis provides valuable insights into the potential impact of the Castle Doctrine on homicide rates, it The analysis provided calculates the Conditional Average Treatment Effect (CATE) by comparing the average homicide rates before and after the implementation of the Castle Doctrine in 2010 for states that implemented the doctrine.It still may not however fully capture the true causal effect due to several potential limitations and assumptions such as:
# - There being no other unmeasured Confounders
# - There being no selection bias
# - Each state has a non-zero probability of implementing or not implementing the Castle Doctrine thus f some states are highly unlikely to implement the doctrine, the generalisability of the results might be limited.
# - Consistency as the treatment effect may not be consistent for all states and the affect of the Castle Doctrine might vary across states due to differences in implementation or other contextual factors.
# 
# 
# 

# One of the fundamental Quasi-Experimental methods of measuring the ATE is the difference in difference approach. It allows us to effectively control for both:
# 
# 1. **Time-Invariant Differences:** Any unobserved characteristics that do not change over time, such as cultural factors or baseline law enforcement practices.
# 2. **Common Temporal Trends:** Broader trends affecting all states, like economic shifts or national policy changes.
# 
# 
#   Mathematically, the **Difference-in-Differences** estimator is expressed as:
# 
#   $DiD = (\overline{Y}_{\text{Post, Treated}} - \overline{Y}_{\text{Pre, Treated}}) - (\overline{Y}_{\text{Post, Control}} - \overline{Y}_{\text{Pre, Control}})$.
# 
# 
# **a)** Plot an **overlaid line plot** of how the **mean of homicide rate** changes across time for states after and before implementing the doctrine. For reference to how your plot should look like, consult **plot.jpg** in your assignment folder.
# 
# 
# 
# **b)**  Report the ATE estimated using the DiD formula where:
# 
# -   Assume the pre-intervention period is 2005, while the post-intervention period is 2010,
# -   Given **S** defines your set of all states, your treatment group **T** consists of the states that have implemented the doctrine in **2010**, and your control group are the states **S\T** in **2005** (i.e. the states apart from those incorporated in T)
# -   Store your result in the `DiD_homicide` variable
# 
# **c)** Comment on the value of ATE obtained through DiD and the approach we used in the previous part. Which one do you believe is more robust to observing an association between homicide rates and the introdution of the Castle Doctrine?

# In[ ]:


# Define the treatment group (states that implemented the doctrine in 2010)
treatment_states = df[(df['post'] == 1) & (df['year'] == 2010)]['sid'].unique()

# Define the control group (states that did not implement the doctrine in 2010)
control_states = df[~df['sid'].isin(treatment_states)]['sid'].unique()

# Calculate the mean homicide rate for each year for both groups
mean_homicide_treatment = df[df['sid'].isin(treatment_states)].groupby('year')['homicide'].mean()
mean_homicide_control = df[df['sid'].isin(control_states)].groupby('year')['homicide'].mean()

# Plot the mean homicide rates over time for both groups
plt.figure(figsize=(10, 6))
plt.plot(mean_homicide_treatment.index, mean_homicide_treatment.values, label='Treatment Group', marker='o')
plt.plot(mean_homicide_control.index, mean_homicide_control.values, label='Control Group', marker='o')
plt.xlabel('Year')
plt.ylabel('Mean Homicide Rate')
plt.title('Mean Homicide Rate Over Time')
plt.legend()
plt.grid(True)
plt.show()

# Calculate the ATE using the DiD formula
pre_treatment_2005 = df[(df['year'] == 2005)]
post_treatment_2010 = df[(df['year'] == 2010)]

mean_pre_treatment_treatment = pre_treatment_2005[pre_treatment_2005['sid'].isin(treatment_states)]['homicide'].mean()
mean_post_treatment_treatment = post_treatment_2010[post_treatment_2010['sid'].isin(treatment_states)]['homicide'].mean()

mean_pre_treatment_control = pre_treatment_2005[pre_treatment_2005['sid'].isin(control_states)]['homicide'].mean()
mean_post_treatment_control = post_treatment_2010[post_treatment_2010['sid'].isin(control_states)]['homicide'].mean()

DiD_homicide = (mean_post_treatment_treatment - mean_pre_treatment_treatment) - (mean_post_treatment_control - mean_pre_treatment_control)

# Print the DiD result
print("DiD_homicide:", DiD_homicide)

# Store the result in the required variable
DiD_homicide


# The DiD approach is more robust than the simple pre-post comparison because it controls for both time-invariant differences and common temporal trends. This helps provide a more accurate estimate of the causal effect of the doctrine on homicide rates.
# 

# Lets Calculate the **Difference-in-Differences (DiD)** estimates for the following crime categories:
# 
# - **Murder**
# - **Robbery**
# - **Larceny**
# - **Assault**
# 
# Using the same DiD formula as previously defined, perform the calculations for each of these crime types to assess whether the implementation of Castle Doctrine laws is associated with a significant change in their respective rates.
# 
# a) **Report the DiD Estimates:**
# 
# b) **Interpret the Results:**
# 

# In[ ]:


# Define the treatment group (states that implemented the doctrine in 2010)
treatment_states = df[(df['post'] == 1) & (df['year'] == 2010)]['sid'].unique()

# Define the control group (states that did not implement the doctrine in 2010)
control_states = df[~df['sid'].isin(treatment_states)]['sid'].unique()

# Calculate the DiD estimates for each crime category

# Murder
pre_treatment_2005 = df[(df['year'] == 2005)]
post_treatment_2010 = df[(df['year'] == 2010)]

mean_pre_treatment_treatment_murder = pre_treatment_2005[pre_treatment_2005['sid'].isin(treatment_states)]['murder'].mean()
mean_post_treatment_treatment_murder = post_treatment_2010[post_treatment_2010['sid'].isin(treatment_states)]['murder'].mean()

mean_pre_treatment_control_murder = pre_treatment_2005[pre_treatment_2005['sid'].isin(control_states)]['murder'].mean()
mean_post_treatment_control_murder = post_treatment_2010[post_treatment_2010['sid'].isin(control_states)]['murder'].mean()

DiD_murder = (mean_post_treatment_treatment_murder - mean_pre_treatment_treatment_murder) - (mean_post_treatment_control_murder - mean_pre_treatment_control_murder)

# Robbery
mean_pre_treatment_treatment_robbery = pre_treatment_2005[pre_treatment_2005['sid'].isin(treatment_states)]['robbery'].mean()
mean_post_treatment_treatment_robbery = post_treatment_2010[post_treatment_2010['sid'].isin(treatment_states)]['robbery'].mean()

mean_pre_treatment_control_robbery = pre_treatment_2005[pre_treatment_2005['sid'].isin(control_states)]['robbery'].mean()
mean_post_treatment_control_robbery = post_treatment_2010[post_treatment_2010['sid'].isin(control_states)]['robbery'].mean()

DiD_robbery = (mean_post_treatment_treatment_robbery - mean_pre_treatment_treatment_robbery) - (mean_post_treatment_control_robbery - mean_pre_treatment_control_robbery)

# Larceny
mean_pre_treatment_treatment_larceny = pre_treatment_2005[pre_treatment_2005['sid'].isin(treatment_states)]['larceny'].mean()
mean_post_treatment_treatment_larceny = post_treatment_2010[post_treatment_2010['sid'].isin(treatment_states)]['larceny'].mean()

mean_pre_treatment_control_larceny = pre_treatment_2005[pre_treatment_2005['sid'].isin(control_states)]['larceny'].mean()
mean_post_treatment_control_larceny = post_treatment_2010[post_treatment_2010['sid'].isin(control_states)]['larceny'].mean()

DiD_larceny = (mean_post_treatment_treatment_larceny - mean_pre_treatment_treatment_larceny) - (mean_post_treatment_control_larceny - mean_pre_treatment_control_larceny)

# Assault
mean_pre_treatment_treatment_assault = pre_treatment_2005[pre_treatment_2005['sid'].isin(treatment_states)]['assault'].mean()
mean_post_treatment_treatment_assault = post_treatment_2010[post_treatment_2010['sid'].isin(treatment_states)]['assault'].mean()

mean_pre_treatment_control_assault = pre_treatment_2005[pre_treatment_2005['sid'].isin(control_states)]['assault'].mean()
mean_post_treatment_control_assault = post_treatment_2010[post_treatment_2010['sid'].isin(control_states)]['assault'].mean()

DiD_assault = (mean_post_treatment_treatment_assault - mean_pre_treatment_treatment_assault) - (mean_post_treatment_control_assault - mean_pre_treatment_control_assault)

# Print the DiD results
print("DiD_murder:", DiD_murder)
print("DiD_robbery:", DiD_robbery)
print("DiD_larceny:", DiD_larceny)
print("DiD_assault:", DiD_assault)

# Store the results in the required variables
DiD_murder
DiD_robbery
DiD_larceny
DiD_assault


# DiD estimates that are positive such as for murder indicate the increase in homicide rates in the treatment group (states that implemented the Castle Doctrine) is greater than the increase in the control group (states that did not implement the Castle Doctrine). This suggests that the Castle Doctrine may be associated with an increase in homicide rates.
# 
# Secondly, most of the other crimes which had negative DiD Estimates which indicates that the increase in those crime rates in the treatment group is less than the increase in the control group thus suggesting that the Castle Doctrine may be associated with a decrease in crime rates for these crimes.

# ###  Multivariate Regression
# 
# We aim to isolate the effect of the Castle Doctrine law on homicide rates, holding all other factors constant. This approach is more robust than simpler comparisons, as it **controls for multiple potential confounding factors** and focuses on within-state changes over time, not just across states.
# 
# 
# 

# In[ ]:


import statsmodels.formula.api as smf

# Our control variables
control_vars = ['unemployrt', 'poverty', 'l_income', 'l_exp_pubwelfare', 'l_police']
controls = ' + '.join(control_vars)

# The OLS model we're using for regression. We use cluster standards error to better adjust for potential within-state correlations.
formula = f'homicide ~ post + {controls} + C(sid) + C(year)'
model = smf.ols(formula=formula, data=df).fit(cov_type='cluster', cov_kwds={'groups': df['sid']})

# Use this summary for your interpretations
display(model.summary())


# 
# 
# a) The coefficient is 0.1402, which represents the effect of the Castle Doctrine law on homicide rates after the law's implementation. This means that the Castle Doctrine law is associated with an estimated 0.1402 increase in homicide rates, holding other variables constant.The earlier DiD approach also yielded a similar estimate which strengthens the robustness of the result.
# 
# b)The p-value for the post variable is 0.683, which is much greater than the 0.05 significance level. This indicates that the coefficient is not statistically significant. This lack of statistical significance means that we cannot confidently say that the Castle Doctrine law had a meaningful impact on homicide rates based on this regression model. The observed effect might be due to random chance rather than a true causal relationship.
# 
# c)  1.  Unemployment (unemployment):
#         Coefficient: -0.0900
#         P-value: 0.450
#         Unemployment appears to have a negative but statistically insignificant relationship with homicide rates. This suggests no robust evidence that changes in unemployment directly affect homicide rates in this model.
#     2.  Poverty (poverty):
#         Coefficient: -0.0530
#         P-value: 0.735
#        Poverty also has a negative but statistically insignificant impact on homicide rates. This indicates that poverty, as included in this model, is not strongly associated with changes in homicide rates.
#     3. Police presence (l_income, as a proxy for law enforcement resources):
#     Coefficient: 0.5178
#     P-value: 0.045
#     Police presence (proxied by logged income) has a statistically significant positive relationship with homicide rates at the 5% level. This might reflect an increase in resources where homicide rates are higher, though further analysis is needed to clarify causation.
# 
# d)The OLS regression has potential limitations, including:
#     - Confounders that we are unaware about
#     - Time and State Fixed Effects
#     - Clustering of Standard Errors
#     - Model Assumptions
#         - Assumes homoscedasticity and a normally distributed error term
#     - Non-Causal Interpretation
#         - This approach still does not fully rule out reverse causality or omitted variable bias, making causal interpretation challenging.
# 
# 
# 
# 
