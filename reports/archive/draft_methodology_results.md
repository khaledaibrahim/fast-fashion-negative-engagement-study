# Draft Methodology and Results

## Study 1 Overview

Study 1 used online customer reviews to explore how low service quality in fast fashion is narrated, emotionally expressed, and associated with negative engagement. The analysis focused on four major brands represented in the review dataset: H&M, ZARA, SHEIN, and Urban Outfitters. The source file contained 5,110 reviews scraped from Trustpilot. For the exploratory stage, the dataset was filtered to 1-3 star reviews only, consistent with the study's focus on low service quality and negative engagement. After filtering short texts and removing exact duplicates by brand, reviewer, and review text, the final exploratory corpus included 4,558 reviews.

## Data Preparation

The pipeline standardized brand names, repaired common text-encoding issues, combined review titles with review bodies, parsed timestamps, and derived monthly review periods for trend analysis. Reviews with fewer than five tokens were excluded to reduce noise. The resulting dataset retained brand, country, rating, review text, review month, and downstream analytic variables including sentiment, aspect markers, emotion markers, and topic assignments.

## Analytic Strategy

The exploratory pipeline combined several transparent text-analytic techniques.

1. Descriptive profiling was used to summarize the filtered sample by brand, star rating, review length, and time period.
2. Sentiment analysis was conducted using VADER compound, positive, neutral, and negative scores.
3. Aspect coding was implemented through a transparent dictionary-based procedure covering delivery/logistics, refunds/returns, customer service, product quality, in-store experience, and trust/reputation.
4. Emotion markers were identified through a parallel dictionary approach covering frustration, disappointment, anger, and disgust.
5. Topic modelling was performed using two complementary models:
   - Latent Dirichlet Allocation (LDA) with six topics and a count-vectorizer representation.
   - BERTopic using a local sklearn embedder based on TF-IDF plus truncated SVD, followed by BERTopic clustering and topic representation.
6. Topic coherence diagnostics were computed using gensim `c_v` coherence scores for both LDA and BERTopic topic-term sets.
7. Brand-level topic, aspect, and emotion distributions were visualized to support transparent comparison across brands.

## Sample Characteristics

The final corpus contained 1,662 H&M reviews, 870 ZARA reviews, 1,017 SHEIN reviews, and 1,009 Urban Outfitters reviews. The sample was heavily concentrated in 1-star complaints, particularly for SHEIN and Urban Outfitters. ZARA reviews were the longest on average, while Urban Outfitters reviews were the shortest.

- H&M: 1,662 reviews, mean review length 99.97 tokens, mean compound sentiment -0.376
- ZARA: 870 reviews, mean review length 122.47 tokens, mean compound sentiment -0.455
- SHEIN: 1,017 reviews, mean review length 70.92 tokens, mean compound sentiment -0.377
- Urban Outfitters: 1,009 reviews, mean review length 44.98 tokens, mean compound sentiment -0.254

## Descriptive and Sentiment Findings

Across brands, the filtered corpus was strongly negative as expected, but there was still meaningful variation in intensity. ZARA exhibited the most negative mean compound sentiment, followed by SHEIN and H&M, while Urban Outfitters was comparatively less negative on average. Negative sentiment labels dominated all four brands, with the highest negative share observed for ZARA (76.6%) and the lowest for Urban Outfitters (57.6%).

The sentiment outputs should be interpreted as intensity markers within an already negative review subset, rather than as general brand sentiment in the marketplace. This distinction is important because the dataset intentionally excludes satisfied and highly positive experiences from the exploratory stage.

## Aspect-Based Findings

The aspect coding suggests that low service quality is not experienced as a single failure type. Instead, it clusters around recurring operational and relational breakdowns.

- Delivery and logistics dominated complaints for H&M (54.6%), SHEIN (46.2%), and Urban Outfitters (53.8%).
- Customer service was the most prominent coded aspect for ZARA (34.5%).
- Refund and return issues were consistently salient across all four brands, especially for SHEIN (20.3%) and ZARA (19.0%).
- Product quality was present but generally secondary to delivery and service complaints.

These patterns suggest that low service quality in fast fashion is perceived both as an operational problem and as a relational problem, supporting the conceptual distinction between simple dissatisfaction and deeper negative engagement.

## Emotion Findings

Frustration was the most prevalent emotion marker across all brands, especially for SHEIN (26.4%) and ZARA (20.0%). Anger and disgust were comparatively more visible in ZARA reviews than in the other brands, while disappointment appeared at similar levels across brands. This pattern is consistent with an escalation account in which repeated or unresolved failures first generate frustration and later develop into more intense affective responses such as anger and disgust.

## LDA Topic Findings

The LDA model produced six interpretable complaint clusters. After term normalization and relabeling, the six publication-facing themes were: In-Store Experience, Exchanges & Returns; Customer Service Failures & Staff Conduct; Product Quality, Size & Wrong Items; Refunds, Contact & Service Recovery; Order Delays, Shipping & Cancellation; and Delivery, Non-Receipt & Parcel Problems. The strongest and most coherent LDA topic centered on order and delivery problems, followed by clusters reflecting store-related experiences, refund and recovery failures, product-quality issues, and broad customer-service breakdowns. At the brand level, H&M was dominated by order delays and shipping disruption, ZARA by in-store experience and customer-service failures, SHEIN by refunds and service-recovery complaints, and Urban Outfitters by non-receipt and parcel problems.

Overall LDA coherence was 0.489, indicating moderate interpretability and giving a stable baseline model for the exploratory stage.

## BERTopic Findings

BERTopic generated more specific semantic complaint clusters than LDA, including:

- manager | receipt | worst customer
- shipped | ship | order online
- shein | evri | reply
- quality | poor quality | cheap
- order cancelled | cancelled order | present
- day delivery | hermes | paid day
- didnt receive | did receive | receive order
- gift | gift card | cards
- wrong item | sent wrong | wrong size

These topics are substantively useful because they surface complaint episodes and failure mechanisms more directly than the broader LDA clusters. For example, BERTopic isolates cancelled orders, courier-specific delivery failures, gift-card disputes, and wrong-item complaints as distinct clusters. However, BERTopic also assigned a relatively large share of reviews to the outlier bucket, particularly for SHEIN and Urban Outfitters. This means BERTopic is valuable as a complementary exploratory lens, but the LDA solution remains the cleaner primary summary model in the current version of the pipeline.

Overall BERTopic coherence was 0.459. Several BERTopic topics performed well on coherence, especially `shipped | ship | order online` (0.539), `order cancelled | cancelled order | present` (0.530), and `shein | evri | reply` (0.524). Some narrower topics had weaker coherence, suggesting that BERTopic is uncovering granular issue clusters that may need manual aggregation into theory-facing constructs.

## Interpretation for the Conceptual Model

Taken together, the exploratory findings suggest the following:

- Low service quality is strongly reflected in delivery, return, and customer-service breakdowns.
- Negative engagement is visible in the language of frustration, anger, disgust, and antagonistic complaint narratives.
- Repetition cues in topics such as refunds, unresolved delivery issues, and order-cancellation problems are consistent with the idea of negative past experience accumulation.
- The complaint structure is not purely transactional; many reviews frame the brand as unreliable, unfair, dismissive, or exploitative, which aligns more closely with the theoretical move from dissatisfaction toward stronger aversive consumer-brand responses.

This supports using Study 1 as an exploratory stage that informs the confirmatory survey model, especially the path from low service quality to negative past experience, brand hate, and eventual brand switching.

## Study 1 to Study 2 Bridge

The role of Study 1 is not to test the structural model directly, but to establish the empirical texture and thematic plausibility of the constructs that are later formalised in Study 2. In that sense, Study 1 functions as an exploratory foundation for the survey stage in three specific ways.

First, Study 1 confirms that low service quality is not a vague background condition but a recurring set of observable failures spanning delivery problems, unresolved refunds, incorrect items, poor product quality, and hostile or ineffective service encounters. This provides practical support for retaining low service quality as the initiating condition in the Study 2 model.

Second, Study 1 shows that some review narratives move beyond isolated dissatisfaction and instead reflect accumulation, repetition, and unresolved histories of failure. These patterns support the inclusion of negative past experience as a distinct construct rather than reducing all negative reactions to one-off dissatisfaction.

Third, the language of active complaint, antagonism, disgust, severe service failure, and explicit bad experiences indicates that consumers do not simply evaluate brands negatively; in some cases they move toward stronger aversive responses consistent with brand hate. This strengthens the logic for distinguishing low service quality and negative past experience from the more severe emotional rejection tested in Study 2.

Accordingly, Study 2 can be positioned as the confirmatory stage that tests whether the pathways surfaced in Study 1 hold at the latent-construct level. The mixed-methods contribution is therefore cumulative: Study 1 identifies and interprets the main forms of low service quality, accumulated negative experience, and emergent aversive response in naturally occurring consumer narratives, while Study 2 tests the formal relationships among low service quality, negative past experience, brand hate, and brand switching.

## Suggested Write-Up Linkage

A concise way to connect the two studies in the manuscript is:

"Study 1 explored naturally occurring low-rated fast-fashion reviews to identify the dominant service-failure themes, emotional expressions, and escalation patterns associated with brand-level complaints. These exploratory findings informed the theoretical refinement and construct emphasis of Study 2, which quantitatively tested the downstream relationships among low service quality, negative past experience, brand hate, and brand switching."

## Recommended Positioning in the Paper

The current evidence supports presenting LDA as the primary exploratory topic model and BERTopic as a complementary robustness and interpretive extension. Aspect coding and emotion coding should be positioned as transparency-oriented analytic layers that help connect inductive topics to the theory-driven constructs used in Study 2.

## Current Limitations

- The exploratory corpus is intentionally restricted to low-rated reviews, so the outputs speak to negative engagement rather than general brand perception.
- SHEIN and Urban Outfitters are almost entirely 1-star in the filtered sample, which limits within-brand rating comparisons.
- BERTopic currently produces a sizeable outlier class, so its outputs should be interpreted as complementary rather than definitive.
- Dictionary-based aspect and emotion coding is transparent and reproducible, but it should be framed as a heuristic layer rather than a perfect classifier.

## Archive Note

This file is retained as an earlier working draft. The current manuscript-facing report is `reports/study1_manuscript_support.md`.
