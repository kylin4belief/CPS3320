# CPS3320 Project
# Group Member: Yasimine, Chloe
Introduction:  	This project is to build a neural network model through Python for Temperature regression prediction.

Data Set:  	The data set we used is the global monthly means temperature. 
		It is from The GISS Surface Temperature Analysis ver. 4, which is an estimate of global surface temperature change. 
		Data sources: https://data.giss.nasa.gov/gistemp/ 

Data Processing:  	In the original dataset, the row is the year and the column is the average temperature of each month. 
		We process the data through the formula provided by csv file, change the monthly average temperature from the original column form to the row form, 
		and add the month note in the row form to clarify the different months.
		We also extracted the monthly average temperature of three months ago, two months ago, one month ago and the current month of each month, and expressed it in the form of columns.
		We chose 384 temperature sample data, and each sample has 5 characteristic value and 1 target value. 
		The first and second columns represent the year and month, respectively.  
		Temp_3 represents the change of average temperature three months ago.
		Temp_2 represents the change of average temperature two months ago.
		Temp1 represents the change of average temperature of the previous month
		The predicted target value is actual, that is, the change of average temperature of the current month.

Input and output: 	By reading the csv file, the system will give the predicted average temperature change value and output it as a diagram. 
		Multiply the obtained data by 100 to get global monthly average temperature in Celsius (deg-C).

Note: 		The zipped folder includes code file, original data set, processed data set, presentation slide, output diagrams and this document.
