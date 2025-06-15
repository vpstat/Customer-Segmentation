The app combines synthetic data generation, machine learning clustering, interactive visualization, and statistical analysis in a single, easy-to-use Streamlit interface. This approach enables both technical and business users to explore customer segmentation and understand key data insights in real time.

## Features

	•	Data is generated with a normal distribution for realism.
	•	All statistical summaries (mean, std, min, max, quartiles, median) are shown in a table.
	•	Interactive prediction and visualization using Plotly, with 3 distinct segment colors.
	•	User input and predicted segment are highlighted.
 
## Steps

1.	Data Generation
	•	Created synthetic customer data for features like Age, Annual Income, and Spending Score.
	•	Used NumPy to generate these features with a normal (Gaussian) distribution to simulate real-world variability.
	•	Applied clipping to ensure values stayed within realistic bounds (e.g., Age between 18 and 70).
2.	Data Preparation
	•	Assembled the generated data into a Pandas DataFrame.
	•	Selected relevant features for clustering and standardized them using `StandardScaler` to ensure equal weighting.
3.	Customer Segmentation (Modeling)
	•	Applied KMeans clustering (with 3 clusters) to segment customers based on their features.
	•	Assigned each customer a segment label.
4.	User Input & Prediction
	•	Built a Streamlit sidebar for users to input their own Age, Annual Income, and Spending Score.
	•	Scaled the user input using the same scaler as the training data.
	•	Predicted the segment for the user input using the trained KMeans model.
5.	Interactive Visualization
	•	Used Plotly Express to create an interactive scatter plot of Annual Income vs. Spending Score, colored by segment.
	•	Added the user’s input as a highlighted marker (gold star) on the plot for easy identification.
	•	Customized the plot with three distinct colors for the segments and adjusted the legend for clarity.
6.	Statistical Analysis
	•	Calculated and displayed descriptive statistics (mean, std, min, max, quartiles, median) for each feature using Pandas `.describe()` and `.median()`.
	•	Presented these statistics in a clear table within the app for user reference.
7.	Segment Profiling
	•	Computed and displayed the average profile (mean values) for each customer segment.
8.	User Experience Enhancements
	•	Provided options to view the raw data.
	•	Organized the app with clear section headers and sidebar controls for a smooth user experience.
