**IDEA**
** Problem Set 2: Quantitative Macroeconomics** 
** Calculation of Labour Share for the whole economy and the Corporate Sector for the US and the UK**
**Mridula Duggal**

import excel "/Users/Mridula/Documents/IDEA/MRes/Year 2/2018-19/Qauntitative Macroeconomics/US_NIPA.xls", sheet("Sheet1") firstrow

/*UNITED STATES OF AMERICA*/
******For the whole US Economy*******
//Renaming the data for convenience
rename Nationalincome Y
rename Compensationofemployees CE
rename ProprietorsincomewithIVAand PI
rename RentalincomeofpersonswithCC RI
rename CorporateprofitswithIVAandC CP
rename Netinterestandmiscellaneousp NI
rename Taxesonproductionandimports Tax
rename LessSubsidies2 subs
rename Businesscurrenttransferpaymen BCTF
rename Currentsurplusofgovernmenten CSG 

//Check the data
br 

//Declaring the time variable
tsset Time 

//Dropping extra data
drop if Y==. 

// Generating the labour share without including PI 
gen theta = CE/(Y-PI)

//Calculation of the Labour share for PI 
gen LS_PI = theta*PI

//Creating WH
gen wh = CE + LS_PI

//Labour share and capital share for the whole US Economy 
gen LS_All = wh/Y
gen RK_All = 1 - LS_All

tsline LS_All

*********For the Corporate Sector*********
//Import the Data
import excel "/Users/Mridula/Documents/IDEA/MRes/Year 2/2018-19/Qauntitative Macroeconomics/Problem Set 2/CS_All.xls", sheet("Sheet1") firstrow

//Renaming the variables
rename Nationalincome Y
rename Corporatebusiness CS_Y
rename Compensationofemployees CS_CE
drop Domesticbusiness

//Generating the time variable
gen time = yearly(Time, "Y")
format time %ty
tsset time
drop Time 

//Creating the labour share 
gen LS_CS = CS_CE/CS_Y

tsline LS_CS


/*UNITED KINGDOM*/
******For the whole UK Economy*******
//Importing the data from the excel file. 

import excel "/Users/Mridula/Documents/IDEA/MRes/Year 2/2018-19/Qauntitative Macroeconomics/Problem Set 2/UK_Compiled.xls", sheet("LS") firstrow clear

//Setting the time variable
gen time = yearly(Time, "Y")
format time %ty
tsset time
drop Time

drop if Y==.

// Generating the labour share without including PI 
gen theta_UK= CE/(Y-PI)

//Calculation of the Labour share for PI 
gen LS_PI_UK = theta_UK*PI

//Creating WH
gen wh_UK = CE + LS_PI_UK

//Labour share and capital share for the whole US Economy 
gen LS_All_UK = wh_UK/Y
gen RK_All_UK = 1 - LS_All_UK

tsline LS_All_UK

*********For the Corporate Sector*********
//Import Data
import excel "/Users/Mridula/Documents/IDEA/MRes/Year 2/2018-19/Qauntitative Macroeconomics/Problem Set 2/UK_Compiled.xls", sheet("LS_CS") firstrow

//Declare Time variable
tsset Time

//Generate Labour Share
gen LS_CS_UK = CS_CE/CS_Y

//Plotting the Labour Share
tsline LS_CS_UK

/*JAPAN*/
******For the whole Japanese Economy*******
//Importing the data from the excel file. 

import excel "/Users/Mridula/Documents/IDEA/MRes/Year 2/2018-19/Qauntitative Macroeconomics/Problem Set 2/UK_Compiled.xls", sheet("LS") firstrow clear

//Setting the time variable
gen time = yearly(Time, "Y")
format time %ty
tsset time
drop Time

drop if Y==.

// Generating the labour share without including PI 
gen theta_UK= CE/(Y-PI)

//Calculation of the Labour share for PI 
gen LS_PI_UK = theta_UK*PI

//Creating WH
gen wh_UK = CE + LS_PI_UK

//Labour share and capital share for the whole US Economy 
gen LS_All_UK = wh_UK/Y
gen RK_All_UK = 1 - LS_All_UK

tsline LS_All_UK

*********For the Corporate Sector*********
//Import Data
import excel "/Users/Mridula/Documents/IDEA/MRes/Year 2/2018-19/Qauntitative Macroeconomics/Problem Set 2/UK_Compiled.xls", sheet("LS_CS") firstrow

//Setting the time variable
gen time = yearly(Time, "Y")
format time %ty
tsset time
drop Time

//Declare Time variable
tsset Time

//Rename variables
rename B1_GIGrossdomesticproducti Y 
rename D1Compensationofemployees CE

//Generate Labour Share
gen LS_JP = CE/Y

//Plotting the Labour Share
tsline LS_CS_JP

