# This file is part of the SOcial Contact RATES (SOCRATES) modelling project
#
# => CONTACT MATRICES FOR COVID-19 MODELLING: BELGIAN SURVEY 2010-2011
#
# Copyright 2020, SIMID, UNIVERSITY OF ANTWERP & HASSELT UNIVERSITY
#___________________________________________________________________________


# clear workspace
rm(list=ls(all=TRUE))
# load help functions
library(shiny)
# load all functions and packages (without messages)
suppressPackageStartupMessages(source('R/socrates_main.R'))
#________________________#
## SETTINGS ####
#________________________#
# set age categories (numeric)
age_breaks <- seq(0,80,10)
# set age categories (text input for SOCRATES)
age_breaks_text <- paste(age_breaks,collapse=',')
# check contact locations
names(opt_location)

# diable "reciprocal matrices"
matrix_weights <- opt_matrix_features[-1]
names(matrix_weights)

# set/check all settings
param_all <- data.frame(country = opt_country[[2]],
daytype = opt_day_type[[6]],
touch = opt_touch[[1]],
duration = opt_duration[[1]],
gender = opt_gender[[1]],
age_breaks_text = age_breaks_text,
telework_reference = 5,
telework_target = 5,
max_part_weight = 3,
bool_transmission_param = FALSE,
age_susceptibility_text = '1',
age_infectiousness_text = '1',
cnt_reduction = 0) # default: no reduction

print(t(param_all))

#________________________#
## ANALYSE ####
#________________________#
# create 3-d array to store all matrices
matrix_all <- array(NA,dim=c(length(age_breaks),length(age_breaks),length(opt_location)+1))
dim(matrix_all)

# loop over all contact locations + "total"
for(i_loc in 1:(length(opt_location)+1)){
# select specific location... or "total
if(i_loc <= length(opt_location)){
sel_location <- opt_location[i_loc]
} else{
sel_location <- opt_location
}
# run SOCRATES-app main function
out_all <- run_social_contact_analysis(country = param_all$country,
daytype = param_all$daytype,
touch = param_all$touch,
duration = param_all$duration,
gender = param_all$gender,
cnt_location = sel_location,
cnt_matrix_features = matrix_weights,
age_breaks_text = param_all$age_breaks_text,
telework_reference = param_all$telework_reference,
telework_target = param_all$telework_target,
max_part_weight = param_all$max_part_weight,
bool_transmission_param = param_all$bool_transmission_param,
age_susceptibility_text = param_all$age_susceptibility_text,
age_infectiousness_text = param_all$age_infectiousness_text,
bool_schools_closed = FALSE,
cnt_reduction = 0)

# add matrix to 3-d array
matrix_all[,,i_loc] = out_all$matrix
}

# add row, column names + locations
rownames(matrix_all) <- rownames(out_all$matrix)
colnames(matrix_all) <- colnames(out_all$matrix)
dimnames(matrix_all)[3] <- list(c(opt_location,Total='Total'))
# save matrices
save(matrix_all, file='contact_matrices_be_2010.RData')
