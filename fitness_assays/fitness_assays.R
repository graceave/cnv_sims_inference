# for Avecilla, Chuong, Li, Sherlock, Gresham, & Ram 2021
# 2021

#ANCOVA analysis to find fitness
library(tidyverse)
setwd("./simulating_evo_paper")
#read in data and add run
bc_props=read_csv('data_props_bc.csv')
Lauer_props=read_csv('data_props_Lauer2018.csv') %>% 
  filter(!(strain %in% c("DGY500", "DGY1315"))) #remove controls

######this function makes a model for a clone and returns the fitness coefficient, se, 95 % CI, using linear modeling#####
make_fitmodel_lm = function(x, data, batch) {
  data = data %>% dplyr::filter(strain == x) %>% dplyr::filter(run == batch)  
  if(nrow(data) == 0) {
    output=NULL
  } else {
    model = lm(transformed~hours, data=data)
    fitness_coef=model$coefficients[[2]]/log(2)
    fitness_se=summary(model)$coefficients[2,2]/log(2)
    pval=summary(model)$coefficients[2,4]
    ci=confint(model, parm='hours', level=0.95)
    output=c(x, fitness_coef, pval, ci[1], ci[2], batch)
  }
  return(output)
}

######this function makes a model for a clone and returns the fitness coefficient, se, 95 % CI, using ancova#####
make_fitmodel_ancova = function(x, data) {
  data = data %>% dplyr::filter(strain %in% 
                                  c(x, 'GAP1_bc_anc1', 'GAP1_bc_anc2','GAP1_bc_anc3')) 
  model = lm(transformed~type*hours+run, data=data)
  fitness_coef=model$coefficients[['typeevo_bc:hours']]/log(2)
  fitness_pval=summary(model)$coefficients['typeevo_bc:hours','Pr(>|t|)']
  ci=confint(model, parm='typeevo_bc:hours', level=0.95)/log(2)
  return(c(x, fitness_coef, fitness_pval, ci[1], ci[2]))
}

#models for Lauer et al clones
l_fit2=map(unique((Lauer_props %>% filter(run == "run2"))$strain), make_fitmodel_lm, 
                data = Lauer_props %>% filter(run == "run2"), batch = "run2")
l_fit3=map(unique((Lauer_props %>% filter(run == "run3"))$strain), make_fitmodel_lm, 
           data = Lauer_props %>% filter(run == "run3"), batch = "run3")
l_fit4=map(unique((Lauer_props %>% filter(run == "run4"))$strain), make_fitmodel_lm, 
           data = Lauer_props %>% filter(run == "run4"), batch = "run4")


l_all_fitness=as_tibble(do.call(rbind, l_fit2)) %>%
  bind_rows(as_tibble(do.call(rbind, l_fit3))) %>%
  bind_rows(as_tibble(do.call(rbind, l_fit4))) 

colnames(l_all_fitness) = c('clone', 'fitness_coef', 'pval', 'ci2.5', 'ci97.5','batch')
l_all_fitness[,2:5]=l_all_fitness[,2:5] %>% mutate_if(is.character, as.numeric)
l_all_fitness = l_all_fitness %>%
  group_by(clone) %>%
  mutate(mean_clone_fit = mean(fitness_coef)) %>% # find avg fitness
  select(c('clone', 'mean_clone_fit')) %>%
  rename(fitness_coef = mean_clone_fit)

#models for all the barcoded clones
bc_clones = unique((bc_props %>% filter(type != "anc_bc"))$strain)
all_fitness=map(bc_clones, make_fitmodel_ancova, data = bc_props)
all_fitness=as_tibble(do.call(rbind, all_fitness))
colnames(all_fitness) = c('clone', 'fitness_coef', 'pval', 'ci2.5', 'ci97.5')
all_fitness[,2:5]=all_fitness[,2:5] %>% mutate_if(is.character, as.numeric)

#combine
all_fitness = all_fitness %>% select(clone, fitness_coef) %>%
  bind_rows(l_all_fitness)

#write out file with fitnesses
write_csv(all_fitness, 'clone_fitness_pergen.csv')


