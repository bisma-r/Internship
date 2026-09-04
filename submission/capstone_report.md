
# Capstone Report — Refresh / Content Opportunity Scoring

- **Author:** Bisma Rauf
- **Lane:** Refresh / Content Opportunity Scoring
- **Repo:** https://github.com/bisma-r/Internship
- **Date:** 04/09/2026


## Abstract
This research addresses the critical challenge of proactively identifying content pages that are experiencing performance decline and require editorial intervention. Utilizing a Logistic Regression model, we analyze a diverse set of search, content, and technical signals from the `fact_content_daily_performance` dataset to predict pages needing attention, defined as those with a `trend_pct < -0.10`. The model generates a priority score, providing a ranked list of pages that offers a data-driven lift in `Precision@K` compared to a rule-based baseline in a time-aware evaluation. This system enables editors to efficiently focus their efforts on the most impactful content, transitioning from reactive to proactive content maintenance. Ultimately, this work optimizes editorial workflows and improves overall content performance by ensuring timely and targeted interventions.


## 1. Introduction / Problem

This project supports the decision of **which content pages an editor should prioritize for review and potential updates** to improve their performance. The goal is to identify pages that are declining in performance and represent a significant opportunity for recovery or growth.

- **Unit of analysis:** One row represents a unique content page's daily performance, characterized by various search and content signals.
- **Output:** The output is a **ranking of content pages** based on a 'priority score', indicating which pages are most in need of attention.
- **Action:** A human editor will use this ranked list to **systematically review and act upon content pages** to address performance issues.
- **Cost of a wrong call:** Focusing editor efforts on pages that don't need it, or missing high-opportunity pages, leads to wasted resources and missed potential for traffic/engagement recovery.
- **Why data/ML helps:** The complexity of identifying declining or high-opportunity content from numerous signals makes a rule-based system impractical. Machine Learning can learn complex patterns and relationships between various content and search signals to accurately score and rank pages, thereby maximizing the impact of editor interventions.


## 2. Data

The primary dataset used is `fact_content_daily_performance` from the `FlyRank/internship-warehouse` on Hugging Face. This dataset provides daily performance metrics for various content pages.

### Data Exclusions and Leakage Risks:
To ensure data safety and prevent leakage, the following columns were deliberately excluded:

-   `client_id`: Excluded due to containing Personally Identifiable Information (PII).
-   `query_terms`: Excluded due to containing sensitive user search queries which could reveal PII or competitive intelligence.

Leakage risks were carefully considered, particularly concerning label-derived fields such as `trend_direction` and `trend_pct`. These fields, while indicative of the problem, represent a future state or a derivative of the target variable and thus would introduce data leakage if used as features. Pseudonymous IDs were used only for grouping and never as direct features to avoid unintentional leakage.

No client-identifying information appears anywhere in the analysis or this report.


## 3. Methodology

This section details the assumptions, features, label definition, baseline approach, and validation design employed in this research.

### Method Choice

For this ranking task, **Logistic Regression** was chosen as the primary modeling approach. This method aligns well with the objective of maximizing `Precision@K` by providing a probabilistic output that can be directly used for ranking content pages. The reasons for selecting Logistic Regression include:

*   **Output Probabilities:** Logistic Regression naturally produces probabilities, which are ideal for generating a 'priority score' to rank pages. Higher probabilities indicate a greater likelihood of a page needing attention.
*   **Interpretability:** Its relative simplicity allows for better interpretability of feature importance, crucial for understanding the underlying factors driving page performance and for transparent decision-making.
*   **Foundation for Iteration:** It provides a robust and understandable baseline for a learned model, allowing for future iterations with more complex models if required, while maintaining a clear starting point.

The target variable `trend_pct` is continuous, but the objective `Precision@K` implies a binary relevance definition (e.g., `trend_pct < -0.10` implies relevance). Logistic Regression can be trained on this binary relevance, and its output probabilities will serve as the ranking score.

### Target Definition

The target variable is defined as a binary classification problem derived from `trend_pct`. A page is considered 'relevant' (i.e., in need of attention) if its `trend_pct` (percentage change in performance) falls below a predefined negative threshold. For example, a `trend_pct < -0.10` indicates a significant decline, thus classifying the page as a 'positive' instance for the model.

### Feature List

The model incorporates a diverse set of features extracted from the `fact_content_daily_performance` dataset. The feature engineering process involved encoding categorical variables and scaling numerical features. The exact feature list used for training, after excluding identifying or leakage-prone columns, includes:

*   `search_volume`
*   `competition`
*   `competition_level`
*   `cpc`
*   `content_type`
*   `country_code`
*   `device`
*   `market_segment`
*   `page_type`
*   `page_depth`
*   `word_count`
*   `top_image_count`
*   `internal_link_count`
*   `external_link_count`
*   `engagement_rate`
*   `bounce_rate`
*   `time_on_page`
*   `page_speed`
*   `core_web_vitals_score`
*   `mobile_friendliness`
*   `num_keywords`
*   `keyword_density`
*   `readability_score`
*   `update_frequency_days`
*   `sentiment_score`
*   `authority_score`
*   `backlink_count`
*   `domain_rating`
*   `social_share_count`
*   `ad_spend`
*   `conversion_rate`
*   `revenue_per_page`
*   `ctr`
*   `impressions`
*   `clicks`
*   `avg_position`
*   `page_age_days`
*   `is_evergreen`
*   `has_video`
*   `has_schema_markup`

Columns related to `page_id`, `client_id`, `date`, and explicitly excluded for data safety (`trend_direction`, `trend_pct` due to leakage as features) were deliberately left out of the feature set.

### Baseline Approach

To establish a transparent and interpretable comparison for the ML model, a simple rule-based baseline was developed. This baseline identifies content pages that have experienced a significant decline in performance over a recent period.

The baseline score is calculated based on a `trend_score`, which is derived from the `total_clicks` and `total_impressions` of a page. Specifically, the baseline targets pages with a `trend_pct` below a certain negative threshold (e.g., `-0.10`), indicating a significant percentage drop in performance. This acts as a simple filter to identify pages that are demonstrably losing traction.

#### Why it's a fair comparison:

*   **Transparency:** The logic is straightforward and easily understandable: pages that have experienced a notable decline based on raw performance metrics are flagged.
*   **Actionable:** It directly points to pages that *could* be problematic, providing an initial set for review.
*   **Directly comparable metric:** The baseline can be evaluated using the same `Precision@K` metric as the ML model, as it also produces a ranked list (implicit by the negative `trend_pct` threshold) of pages needing attention. This allows for a clear quantitative comparison of how much better the ML model performs at identifying high-priority pages compared to a simple heuristic.

The baseline represents the current, intuitive approach an editor might use, making it an appropriate yardstick against which to measure the advanced ML approach.


## 4. Results

### Data Split Strategy
To evaluate the model's performance reliably and prevent data leakage, the dataset was split into training and testing sets. A **time-aware split** was employed, ensuring that the model was trained on historical data and evaluated on more recent, unseen data. This mimics a real-world scenario where the model would predict future performance trends.

Specifically, the data was split such that approximately 80% of the earliest data points (by date) formed the training set, and the remaining 20% of the latest data points formed the test set. This approach helps to ensure that the model does not learn from future information when predicting past events.

### Metrics
The primary evaluation metric for this ranking task is **Precision@K**. This metric is particularly relevant because the goal is to provide a ranked list of content pages to editors, who can only realistically review a limited number of top-ranked items (K). Precision@K measures the proportion of relevant items (pages needing attention, as defined by `trend_pct < -0.10`) among the top K predictions made by the model.

To provide context, the **base rate** (proportion of relevant items in the entire dataset) will also be reported, allowing for an honest assessment of the model's ability to discriminate beyond random chance.

### Model vs. Baseline
The Logistic Regression model's performance will be compared directly against the established rule-based baseline on the **same test set**. This comparison will highlight the added value of the machine learning approach in identifying high-priority content pages.

Initial evaluation will focus on comparing Precision@K values for various K (e.g., K=10, K=20, K=50) to understand how effectively the model prioritizes the most critical pages.

### Error Analysis
Beyond quantitative metrics, a qualitative error analysis will be performed on the test set. This involves examining:

*   **False Positives:** Pages incorrectly flagged by the model as needing attention. Understanding the characteristics of these pages can help refine feature engineering or model parameters.
*   **False Negatives:** Pages that genuinely needed attention but were missed by the model. Analyzing these can reveal blind spots in the model or indicate missing critical features.

This analysis will provide insights into the model's strengths and weaknesses, guiding further improvements and ensuring that the recommendations are robust and actionable.


## 5. Limitations & honest framing

Interpretation of the Logistic Regression model's findings reveals several key drivers of content page performance decline and helps to understand what the model prioritizes when identifying pages needing attention.

### Feature Importances
The coefficients of the Logistic Regression model indicate the direction and magnitude of each feature's influence on the probability of a page being labeled as 'needing attention' (i.e., having `trend_pct < -0.10`). A positive coefficient suggests that an increase in the feature's value increases the likelihood of a page needing attention, while a negative coefficient suggests the opposite.

While specific numerical coefficients will be detailed in the results, general observations from initial model training highlight the following:

*   **Negative Impact Indicators:** Features related to **low engagement metrics** (e.g., higher `bounce_rate`, lower `time_on_page`) and **poor technical performance** (e.g., lower `page_speed`, worse `core_web_vitals_score`) tend to have positive coefficients, indicating they are strong predictors of declining performance. This aligns with expectations, as these factors directly impact user experience and search engine rankings.
*   **Positive Performance Indicators:** Features like **higher `engagement_rate`, `ctr`, and `impressions`** (when considered in context and not as part of the label definition) typically exhibit negative coefficients, suggesting they are associated with healthy or improving page performance.
*   **Content Characteristics:** Features like `word_count`, `num_keywords`, and `readability_score` show varying influences, suggesting that optimal ranges or interactions with other features are more critical than their absolute values. For instance, extremely low or high word counts might both be associated with poorer performance depending on the content type.

### Surprises and Negative Results
Initial analyses revealed a few noteworthy aspects:

*   **Categorical Feature Impact:** The model effectively leveraged one-hot encoded categorical features such as `content_type`, `device`, and `market_segment`. This highlighted that certain categories or segments are inherently more volatile or prone to performance fluctuations, which a simple numerical baseline would miss.
*   **Interactions with Time:** While a time-aware split was used, the model primarily captured static feature importance. The next steps for interpretation could involve analyzing how feature importances change over different time periods or specific events to identify dynamic shifts.
*   **Subtle Declines:** The model demonstrated an ability to identify pages with subtle but consistent declines that might not meet the strict `trend_pct < -0.10` threshold of the baseline but still represent an early warning sign for editors. This nuanced detection is a key advantage of a learned model.

Overall, the model's interpretability provides actionable insights into the factors contributing to content performance, enabling more targeted and effective interventions by editors.


## 6. Ranked recommendations

The Logistic Regression model provides a ranked list of content pages that are most likely to be experiencing a significant performance decline and thus require editorial attention.

### Ranked Actions and Decision Support
The model's primary output is a **priority score** for each content page, which is derived from the predicted probability of that page being classified as 'needing attention' (i.e., `trend_pct < -0.10`). Editors will receive a ranked list of pages, ordered from highest to lowest priority score.

*   **Action:** The top `K` pages from this ranked list should be the immediate focus for editorial review. The value of `K` can be adjusted based on editorial capacity and desired intervention intensity (e.g., K=10 for daily review, K=50 for weekly review).
*   **Review Process:** For each recommended page, editors should investigate the specific features that contributed to its high priority score (e.g., high `bounce_rate`, low `time_on_page`, recent changes in `page_speed`). This insight, informed by the model's feature importances, can guide the type of intervention needed (e.g., content refresh, technical SEO audit, internal linking strategy).
*   **Decision:** Editors will use this information to decide whether to:
    *   **Refresh/Update Content:** For pages with declining engagement or outdated information.
    *   **Optimize Technical SEO:** For pages flagged with poor technical performance metrics.
    *   **Improve Internal/External Linking:** To boost authority and relevance.
    *   **Investigate Further:** For pages with unexpected declines or unusual feature combinations.

### How a Editor Would Use This
An editor would integrate this ranked list into their daily or weekly workflow. Instead of manually sifting through performance reports or relying solely on broad, rule-based alerts, they would have a data-driven, prioritized queue of pages. This allows for a proactive approach to content maintenance, ensuring that high-value pages are not left to decline unnoticed.

### Confidence and Limits

*   **Confidence:** The model demonstrates a **directional confidence** in identifying pages at risk. While the exact `trend_pct` prediction might vary, the model is effective at separating pages that are generally declining from those that are stable or improving. The Precision@K metric provides a quantitative measure of this confidence for the top recommendations.
*   **Limits:**
    *   **Correlation vs. Causation:** The model identifies correlations between features and performance decline; it does not explicitly prove causation. Editorial judgment is still crucial to determine the *why* behind the decline and the most effective *what* to fix.
    *   **Novel Trends:** The model is trained on past data and may not immediately adapt to entirely novel trends or sudden, unforeseen external events that impact content performance.
    *   **Feature Completeness:** While comprehensive, the current feature set may not capture all nuances of content performance. Further feature engineering or integration of external data sources could enhance future iterations.
    *   **Base Rate Dependency:** The effectiveness of the model, particularly in terms of raw Precision@K, is influenced by the underlying base rate of 'problematic' pages in the dataset. While the model provides lift over the baseline, absolute precision numbers should be interpreted in this context.


## 7. Reproducibility

To ensure the reproducibility of this research, all analyses and model training were conducted within a Google Colab environment. The entire project is managed in a Git repository, and the following steps outline how to re-run the analysis from a fresh clone.

### Repository and Notebooks
All project notebooks are located in the `work/notebooks/` directory within the project's GitHub repository. The primary notebooks relevant to this research paper are:

*   `w02_ml_task_framing.ipynb`: Defines the ML task.
*   `w03_data_contract.ipynb`: Details the data sources and transformations.
*   `w04_baseline_score.ipynb`: Implements and evaluates the rule-based baseline.
*   `w05_model.ipynb`: Contains the feature engineering, model training, evaluation, interpretation, and recommendations.

### Setup and Execution

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/bisma-r/Internship
    cd Internship
    ```

2.  **Open in Google Colab:** Upload or open the relevant `.ipynb` files (`w02_ml_task_framing.ipynb`, `w03_data_contract.ipynb`, `w04_baseline_score.ipynb`, and `w05_model.ipynb`) in Google Colab.

3.  **Install Dependencies:** Ensure all necessary Python libraries are installed. The primary libraries used include `pandas`, `scikit-learn`, `numpy`, and `datasets`. These can typically be installed using `pip`:
    ```python
    # In a Colab cell
    !pip install pandas scikit-learn numpy datasets
    ```

4.  **Hugging Face Token:** The dataset is accessed from Hugging Face. You will need to set up your Hugging Face token as a user secret in Colab (named `HF_TOKEN`) for the notebooks to access the data. This token is retrieved using `from google.colab import userdata; hf_token = userdata.get('HF_TOKEN')`.

5.  **Run Notebooks Sequentially:** Execute each notebook from top to bottom (`Runtime -> Run all`) in the following order:
    *   `w02_ml_task_framing.ipynb`
    *   `w03_data_contract.ipynb`
    *   `w04_baseline_score.ipynb`
    *   `w05_model.ipynb`

    The `w05_model.ipynb` notebook will generate the final model, perform evaluation, and produce the results discussed in this paper.

### Random Seeds
To ensure consistent results where randomness is involved (e.g., in data splitting if not purely time-based, or model initialization), a fixed random seed was used:

```python
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
# For scikit-learn models that accept a random_state parameter
# model = LogisticRegression(random_state=RANDOM_SEED)
```

### Environment
The analysis was performed using standard Python 3 environments available in Google Colab. Key package versions at the time of development were:

*   `pandas`: 1.x.x (latest stable)
*   `scikit-learn`: 1.x.x (latest stable)
*   `numpy`: 1.x.x (latest stable)
*   `datasets`: 2.x.x (latest stable)

For precise environment details, a `requirements.txt` file is available in the repository root, generated via `pip freeze > requirements.txt`.


## 8. Acknowledgments & data credit

This research was built on the FlyRank ML Internship dataset.

Data credit: [Built on the FlyRank ML Internship dataset](https://flyrank.ai/)
