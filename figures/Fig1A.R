# figure 1a
# avecilla et al


library(tidyverse)
setwd("/Volumes/GoogleDrive/.shortcut-targets-by-id/19NQRq-XLPpjqJGmLbP4MHRjala2WVahk/Avecilla et al Simulating CNV evolution/scripts/cnv_sim_inf_old")

pops = read_csv('./hpc/new/PopPropForABC_all.csv') 
gens = read_csv('./hpc/generations.csv')

data = cbind(pops, gens) %>% 
  pivot_longer(-Generation, names_to="Population", values_to="PropCNV")

mycolors = c("bc01"="#a6cee3",
             "bc02"="#1f78b4",
             "gln_01"="#b2df8a",
             "gln_02"="#33a02c",
             "gln_03"="#fb9a99",
             "gln_04"="#e31a1c",
             "gln_05"="#fdbf6f",
             "gln_06"="#ff7f00",
             "gln_07"="#cab2d6",
             "gln_08"="#6a3d9a",
             "gln_09"="#ffed6f")

p = ggplot(data, 
       aes(Generation, PropCNV, color=Population)) +
  geom_line(size=1.5) + 
  scale_color_manual(values=mycolors) +
  theme_classic(base_size=20) +
  scale_x_continuous(expand = c(0, 0), breaks = c(50, 100, 150, 200, 250)) + 
  scale_y_continuous("Proportion of cells with GAP1 CNV", expand = c(0, 0))

ggsave('../../figures/Fig1A.pdf', p, width = 7.2, height=5, units = "in")


#get supplementary data from Lauer et al into the right format
bc = read_csv("~/Downloads/SLGating03202018_GADATA.csv")

bc = bc %>% filter(Sample %in% c("gln_bc01", "gln_bc02")) %>%
  filter(Generation != 174) %>%
  select(Generation, Sample, PopProp_CNV) %>%
  pivot_wider(names_from = Sample, values_from = PopProp_CNV) %>%
  rename(bc01=gln_bc01, bc02 = gln_bc02)

write_csv(bc %>% select(bc01, bc02), './hpc/PopPropForABC_bc.csv')
write_csv(bc %>% select(Generation), './hpc/generations_bc.csv')
