🔫 Analyzing the Impact of Castle Doctrine Laws on Violent Crime Rates in the USA
This repository contains a data-driven analysis of the relationship between the implementation of Castle Doctrine laws and violent crime rates across U.S. states. The notebook explores whether these self-defense laws correlate with any significant increase or decrease in violent crimes using empirical data and visualization techniques.

📘 Project Description


Castle Doctrine laws permit individuals to use force—often deadly force—in defense of themselves or their property without the duty to retreat. This project investigates the potential causal or correlative impact of these laws on violent crime statistics such as:

Homicide

Aggravated assault

Robbery

Overall violent crime

This study uses before-and-after comparisons and cross-state data to determine trends and shifts in crime patterns post-legislation.


📊 Key Features
Data Acquisition & Preprocessing:

Loads crime rate data for U.S. states over multiple years.

Marks states with Castle Doctrine enactment years.

Handles null values and ensures time alignment across datasets.

Exploratory Data Analysis (EDA):

Time-series analysis of crime rate trends.

Comparison of crime rates before and after law implementation.

Aggregated views of crime rate changes across Castle vs non-Castle states.

Visualizations:

Line plots to compare trends pre- and post-law.

Bar charts for average changes in crime rate.

Highlighted outlier states and post-enactment shifts.

Policy Implications:

Discusses limitations and potential interpretations of results.

Lays groundwork for further statistical modeling (e.g., difference-in-differences, regression).

🛠️ Tools & Technologies
Python 3.x

Jupyter Notebook

Libraries:

pandas

matplotlib

seaborn

numpy

🚀 Getting Started
Clone the repository:

'''bash
Copy code
git clone https://github.com/yourusername/castle-doctrine-analysis.git
cd castle-doctrine-analysis
Install dependencies:

'''bash
Copy code
pip install pandas matplotlib seaborn numpy
Launch the notebook:

'''bash
Copy code
jupyter notebook "Analyzing_the_Impact_of_Castle_Doctrine_Laws_on_Violent_Crime_Rates_in_the_USA.ipynb"
Run all cells to reproduce the full analysis.

📈 Sample Insights
Some states show a rise in violent crime rates post-legislation, while others remain stable.

Homicide and assault rates exhibit the most notable variation across states.

The analysis opens up important questions about public safety, gun control, and self-defense laws.

🧠 Future Work
Incorporate socioeconomic and demographic control variables.

Apply statistical significance testing (e.g., t-tests, regression models).

Build an interactive dashboard to explore individual state timelines.

👤 Author
Rafay Salim
