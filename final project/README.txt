Introduction:  	This project is to build a neural network model through Python for Temperature regression prediction.

Data Set:  	The data set we used is the global monthly means temperature. 
		It is from The GISS Surface Temperature Analysis ver. 4, which is an estimate of global surface temperature change. 

Data Processing:  	We chose 384 temperature sample data, and each sample has 5 characteristic value and 1 target value. 
		The first and second columns represent the year and month, respectively.  
		Temp_3 represents the change of average temperature three months ago.
		Temp_2 represents the change of average temperature two months ago.
		Temp1 represents the change of average temperature of the previous month
		The predicted target value is actual, that is, the change of average temperature of the current month.

Input and output: 	By reading the csv file, the system will give the predicted average temperature change value and output it as a diagram. 
		Multiply the obtained data by 100 to get global monthly average temperature in Celsius (deg-C).

Note: 		The zipped folder includes code file, original data set, processed data set, presentation slide, output diagrams and this document.