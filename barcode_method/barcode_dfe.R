# barcode lineage dfe

library(tidyverse)
library(groupdata2)

cnv_barcodes = read_csv("CNV_barcodes.csv") %>%
  mutate(CNV = "CNV")

bc02 = read_csv('bc02_cluster.csv') 

bc_02_filtered = bc02 %>% 
  #select only lineages with >0 reads in first time points
  filter(time_point_1 > 0) %>%
  #select lineages that don't go to 0 then over zero
  filter(!(time_point_2 == 0 & time_point_3 > 0)) %>%
  filter(!(time_point_3 == 0 & time_point_4 > 0)) %>%
  filter(!(time_point_4 == 0 & time_point_5 > 0)) %>%
  filter(!(time_point_5 == 0 & time_point_6 > 0)) %>%
  mutate(population = "bc02")

bc_02_freq = bc_02_filtered %>%
  #convert from reads to relative frequency
  mutate(time_point_1 = time_point_1/sum(time_point_1),
         time_point_2 = time_point_2/sum(time_point_2),
         time_point_3 = time_point_3/sum(time_point_3),
         time_point_4 = time_point_4/sum(time_point_4),
         time_point_5 = time_point_5/sum(time_point_5),
         time_point_6 = time_point_6/sum(time_point_6)) %>%
  pivot_longer(cols=starts_with("time"), names_to = "generation", values_to = "relative_freq") %>%
  mutate(generation = case_when(generation == "time_point_1" ~ 5,
                                generation == "time_point_2" ~ 37,
                                generation == "time_point_3" ~ 62,
                                generation == "time_point_4" ~ 99,
                                generation == "time_point_5" ~ 124,
                                generation == "time_point_6" ~ 149)) %>%
  mutate(cell_number = relative_freq*3e8)

bc_02_count = bc_02_filtered %>%
  pivot_longer(cols=starts_with("time"), names_to = "generation", values_to = "reads") %>%
  mutate(generation = case_when(generation == "time_point_1" ~ 5,
                                generation == "time_point_2" ~ 37,
                                generation == "time_point_3" ~ 62,
                                generation == "time_point_4" ~ 99,
                                generation == "time_point_5" ~ 124,
                                generation == "time_point_6" ~ 149))

ggplot(bc_02_freq, aes(generation, cell_number, group=Center)) +
  geom_line(alpha=0.1) + 
  scale_y_log10(limits = c(1, 1e8), c(10^0, 10^1, 10^2, 10^3, 10^4, 10^5, 10^6, 10^7, 10^8))

ggplot(bc_02_count, aes(reads,group=generation,color=as.factor(generation))) +
  geom_density() +
  scale_x_log10()

write_csv(bc_02_filtered %>% select(-Cluster.ID,-Center, -Cluster.Score, -population), file = "bc_02_filtered_PyFit.csv", col_names = F)

bc04 = read_csv('bc04_cluster.csv') 

bc_04_filtered = bc04 %>% 
  #select only lineages with >0 reads in first time points
  filter(time_point_1 > 0) %>% #& time_point_2 > 0 & time_point_3 > 0) %>%
  #select lineages that don't go to 0 then over zero
  filter(!(time_point_2 == 0 & time_point_3 > 0)) %>%
  filter(!(time_point_3 == 0 & time_point_4 > 0)) %>%
  filter(!(time_point_4 == 0 & time_point_5 > 0)) %>%
  filter(!(time_point_5 == 0 & time_point_6 > 0)) %>%
  mutate(population = "bc01") #bc04 is bc01 in Lauer et al
bc_04_freq = bc_04_filtered %>%
  #convert from reads to relative frequency
  mutate(time_point_1 = time_point_1/sum(time_point_1),
         time_point_2 = time_point_2/sum(time_point_2),
         time_point_3 = time_point_3/sum(time_point_3),
         time_point_4 = time_point_4/sum(time_point_4),
         time_point_5 = time_point_5/sum(time_point_5),
         time_point_6 = time_point_6/sum(time_point_6)) %>%
  pivot_longer(cols=starts_with("time"), names_to = "generation", values_to = "relative_freq") %>%
  mutate(generation = case_when(generation == "time_point_1" ~ 5,
                                generation == "time_point_2" ~ 37,
                                generation == "time_point_3" ~ 62,
                                generation == "time_point_4" ~ 99,
                                generation == "time_point_5" ~ 124,
                                generation == "time_point_6" ~ 149)) %>%
  mutate(cell_number = relative_freq*3e8)


ggplot(bc_04_freq, aes(generation, cell_number, group=Center)) +
  geom_line(alpha=0.1) + 
  scale_y_log10(limits = c(1, 1e8), c(10^0, 10^1, 10^2, 10^3, 10^4, 10^5, 10^6, 10^7, 10^8))

write_csv(bc_04_filtered %>% select(-Cluster.ID,-Center, -Cluster.Score, -population), file = "bc_04_filtered_PyFit.csv", col_names = F)

alldata = bc_02_filtered %>% 
  bind_rows(bc_04_filtered) %>%
  select(-Cluster.ID, -Cluster.Score) %>%
  left_join(cnv_barcodes, by= c("Center" = "barcode", "population"="population")) %>% 
  mutate(time_point_1 = time_point_1/sum(time_point_1),
         time_point_2 = time_point_2/sum(time_point_2),
         time_point_3 = time_point_3/sum(time_point_3),
         time_point_4 = time_point_4/sum(time_point_4),
         time_point_5 = time_point_5/sum(time_point_5),
         time_point_6 = time_point_6/sum(time_point_6)) %>%
  pivot_longer(cols=starts_with("time"), names_to = "generation", values_to = "relative_freq") %>%
  mutate(generation = case_when(generation == "time_point_1" ~ 5,
                                generation == "time_point_2" ~ 37,
                                generation == "time_point_3" ~ 62,
                                generation == "time_point_4" ~ 99,
                                generation == "time_point_5" ~ 124,
                                generation == "time_point_6" ~ 149)) %>%
  mutate(cell_number = relative_freq*3e8) %>%
  mutate(CNV = if_else(is.na(CNV), "noCNV", "CNV"))

ggplot(alldata, aes(generation, cell_number, group=Center, color=CNV)) +
  geom_line(alpha=0.05) + 
  scale_y_log10("Number of cells", limits = c(1, 1e8), c(10^0, 10^1, 10^2, 10^3, 10^4, 10^5, 10^6, 10^7, 10^8)) +
  facet_wrap(~population) + 
  theme_minimal(base_size = 15) +
  scale_color_manual(values = c("CNV" = "#6DB966", "noCNV" = "black")) 

# plot how many barcodes are CNV vs not CNV at each time point (bar plot)
ggplot(alldata %>% filter(cell_number >0), aes(generation, fill = CNV)) +
  geom_bar(position = "dodge") +
  facet_wrap(~population) +
  scale_fill_manual(values = c("CNV" = "#6DB966", "noCNV" = "black")) +
  scale_y_log10() +
  theme_minimal(base_size = 15) +
  theme(panel.spacing.x=unit(3, "lines"))


ggplot(alldata %>% filter(cell_number > 10^3 & cell_number < 10^4), aes(generation, cell_number,group=Center)) +
  geom_line(alpha = 0.05)

alldata = alldata %>%
  left_join(alldata %>% 
    group_by(population, Center) %>%
    filter(generation == 149 & cell_number > 0) %>%
    mutate(survival = "survives") %>%
      select(Center, population, survival)) 

alldata = alldata %>%
  mutate(survival = if_else(is.na(survival),"extinct", "survives"))

ggplot(alldata, aes(generation, cell_number, group=Center, color=CNV, alpha = survival)) +
  geom_line() + 
  scale_y_log10("Number of cells", limits = c(1, 1e8), c(10^0, 10^1, 10^2, 10^3, 10^4, 10^5, 10^6, 10^7, 10^8)) +
  facet_wrap(~population) + 
  theme_minimal(base_size = 15) +
  scale_color_manual(values = c("CNV" = "#6DB966", "noCNV" = "black")) +
  scale_alpha_manual(values = c(0.05, 0.13))#, guide = FALSE)


## rough estimate of DFE ###
# try fitting regression to each barcode trajectory
# if it goes to zero take out points after the first zero

alldata_wide = bc_02_filtered %>% 
  bind_rows(bc_04_filtered) %>%
  select(-Cluster.ID, -Cluster.Score) %>%
  left_join(cnv_barcodes, by= c("Center" = "barcode", "population"="population")) %>% 
  mutate(time_point_1 = time_point_1/sum(time_point_1),
         time_point_2 = time_point_2/sum(time_point_2),
         time_point_3 = time_point_3/sum(time_point_3),
         time_point_4 = time_point_4/sum(time_point_4),
         time_point_5 = time_point_5/sum(time_point_5),
         time_point_6 = time_point_6/sum(time_point_6)) %>%
  filter(time_point_2 > 0)

for(i in 1:nrow(alldata_wide)) {
  t = alldata_wide[i,]
  if(t$time_point_3[1] == 0) {
    t=t %>% select(-time_point_4,-time_point_5,-time_point_6)
  } else if(t$time_point_4[1] == 0) {
    t=t %>% select(-time_point_5,-time_point_6)
  }else if(t$time_point_5[1] == 0) {
    t=t %>% select(-time_point_6)
  }
  t = t%>%  
    pivot_longer(cols=starts_with("time"), names_to = "generation", values_to = "relative_freq") %>%
    mutate(generation = case_when(generation == "time_point_1" ~ 5,
                                  generation == "time_point_2" ~ 37,
                                  generation == "time_point_3" ~ 62,
                                  generation == "time_point_4" ~ 99,
                                  generation == "time_point_5" ~ 124,
                                  generation == "time_point_6" ~ 149))
    alldata_wide$fitness[i] = lm(relative_freq~generation, data=t)$coefficients['generation']
}
 
#### DOWNSAMPLING ####

bc_02_downsampled = bc_02_filtered %>%
  pivot_longer(cols=starts_with("time"), names_to = "timepoint", values_to = "reads") %>%
  uncount(reads) %>%
  downsample(cat_col = "timepoint") %>%
  group_by(timepoint, Center) %>%
  summarize(bc_count = n()) %>%
  pivot_wider(names_from = timepoint, values_from = bc_count) %>%
  replace(is.na(.), 0)

#write for pyfitmut
write_csv(bc_02_downsampled %>% select(-Center), file = "bc_02_downsampled_PyFit.csv", col_names = F)

bc_02_downsampled %>%
  #convert from reads to relative frequency
  mutate(time_point_1 = time_point_1/sum(time_point_1),
         time_point_2 = time_point_2/sum(time_point_2),
         time_point_3 = time_point_3/sum(time_point_3),
         time_point_4 = time_point_4/sum(time_point_4),
         time_point_5 = time_point_5/sum(time_point_5),
         time_point_6 = time_point_6/sum(time_point_6)) %>%
  pivot_longer(cols=starts_with("time"), names_to = "generation", values_to = "relative_freq") %>%
  mutate(generation = case_when(generation == "time_point_1" ~ 5,
                                generation == "time_point_2" ~ 37,
                                generation == "time_point_3" ~ 62,
                                generation == "time_point_4" ~ 99,
                                generation == "time_point_5" ~ 124,
                                generation == "time_point_6" ~ 149)) %>%
  mutate(population = "bc02") %>%
  mutate(cell_number = relative_freq*3e8) %>%
  ggplot(aes(generation, cell_number, group=Center)) +
  geom_line(alpha=0.1) + 
  scale_y_log10(limits = c(1, 1e8), c(10^0, 10^1, 10^2, 10^3, 10^4, 10^5, 10^6, 10^7, 10^8))

bc_04_downsampled = bc04 %>%
  pivot_longer(cols=starts_with("time"), names_to = "timepoint", values_to = "reads") %>%
  uncount(reads) %>%
  downsample(cat_col = "timepoint") %>%
  group_by(timepoint, Center) %>%
  summarize(bc_count = n()) %>%
  pivot_wider(names_from = timepoint, values_from = bc_count) %>%
  replace(is.na(.), 0)

#write for pyfitmut
write_csv(bc_04_downsampled %>% select(-Center), file = "bc_04_downsampled_PyFit.csv", col_names = F)


bc_04_downsampled %>%
  #convert from reads to relative frequency
  mutate(time_point_1 = time_point_1/sum(time_point_1),
         time_point_2 = time_point_2/sum(time_point_2),
         time_point_3 = time_point_3/sum(time_point_3),
         time_point_4 = time_point_4/sum(time_point_4),
         time_point_5 = time_point_5/sum(time_point_5),
         time_point_6 = time_point_6/sum(time_point_6),
         time_point_7 = time_point_7/sum(time_point_7)) %>%
  pivot_longer(cols=starts_with("time"), names_to = "generation", values_to = "relative_freq") %>%
  mutate(generation = case_when(generation == "time_point_1" ~ 0,
                                generation == "time_point_2" ~ 5,
                                generation == "time_point_3" ~ 37,
                                generation == "time_point_4" ~ 62,
                                generation == "time_point_5" ~ 99,
                                generation == "time_point_6" ~ 124,
                                generation == "time_point_7" ~ 149)) %>%
  mutate(population = "bc02") %>%
  mutate(cell_number = relative_freq*3e8) %>%
  ggplot(aes(generation, cell_number, group=Center)) +
  geom_line(alpha=0.1) + 
  scale_y_log10(limits = c(1, 1e8), c(10^0, 10^1, 10^2, 10^3, 10^4, 10^5, 10^6, 10^7, 10^8))


######other#####
bc_new = props %>% pivot_longer(cols = !barcode, names_to = "library", values_to = "barcode_prop") %>%
  separate(library, into=c(NA, "pop", "generation"), sep="_") %>%
  mutate(generation = str_remove(generation, "g"))
bc_grace = bc_new %>% 
  pivot_wider(names_from = generation, values_from = barcode_prop) %>% 
  dplyr::filter(!is.na(`37`)) %>% select(-`5`) %>%
  mutate(cont_bc = case_when(!is.na(`37`) & !is.na(`62`) & 
                               !is.na(`99`) & !is.na(`124`) & !is.na(`149`) ~ "all",
                             !is.na(`37`) & !is.na(`62`) & 
                               !is.na(`99`) & !is.na(`124`) ~ "first_4_timepts",
                             !is.na(`37`) & !is.na(`62`) & 
                               !is.na(`99`) & !is.na(`149`) ~ "skip124",
                             !is.na(`37`) & !is.na(`62`) & 
                               !is.na(`99`) ~ "first_3_timepts",
                             !is.na(`37`) & !is.na(`62`) & 
                               !is.na(`124`) & !is.na(`149`) ~ "skip99",
                             !is.na(`37`) & !is.na(`62`) ~ "first_2_timepts",
                             !is.na(`37`) & 
                               !is.na(`99`) & !is.na(`124`) & !is.na(`149`) ~ "skip62",
                             !is.na(`37`) ~ "37only")) %>% 
  mutate_if(is.numeric, funs(replace_na(., 0))) %>%
  mutate(one_perc = if_else(`37` > 0.01 | `62` > 0.01 | `99` > 0.01 | `124` > 0.01 | `149` > 0.01, "greater_1", "less_1")) %>%
  pivot_longer(cols = c(-barcode, -pop, -cont_bc, -one_perc), names_to = "generation", values_to = "proportion")
for_plot_levyfig1 = bc_grace %>% 
  dplyr::filter(!(cont_bc == "skip62" & generation == 62),
                !(cont_bc == "skip99" & generation == 99),
                !(cont_bc == "skip124" & generation == 124), !(generation==32 & proportion == 0)) %>%
  mutate(generation = as.numeric(generation), cell_num = proportion*3e8)
ggplot(for_plot_levyfig1, aes(generation, cell_num, group = barcode)) +
  geom_line(alpha=0.5) +
  theme_classic() +
  facet_wrap(~pop) +
  scale_y_log10()
for_plot_lauerfig5 = bc_grace %>% 
  mutate(color_cat = if_else(one_perc == "less_1", "less_one", barcode)) %>%
  group_by(color_cat, generation, pop) %>%
  summarise(proportion_forplot = sum(proportion))
ggplot(for_plot_lauerfig5, aes(x=as.numeric(generation), y=proportion_forplot, fill=color_cat)) + 
  #geom_bar(aes(fill = color_cat), stat = "identity", show.legend = F) +
  geom_area(alpha=0.6 , size=1, colour="black", show.legend = F) +
  theme_classic()+
  scale_y_continuous(expand=c(0,0), limits=c(0,1)) +
  #scale_fill_manual(values = sample(mycolors)) +
  facet_wrap(~ pop)