# inference on real data
# g avecilla


library(tidyverse)
library(ggbeeswarm)
library(patchwork)
theme_set(theme_bw(base_size = 15))
dir = "/Volumes/GoogleDrive/My Drive/Gresham Lab_Grace/cnv_sim_Ram_collab/Final_versions_Feb21/est_real_params/Lauer"
setwd(dir)

files = list.files(path = dir, pattern = '*est_real_params.csv')
#files = c(files[str_detect(files, "Lauer")], 
 #         list.files(path = dir, pattern = '*est_real_params_116g.csv'))

####EDIT TO READ IN PYABC FILES ALSO####
read_est_params = function(x) {
  if(str_detect(x, "SNPE")==TRUE){
    file = read_csv(x, col_names = c("Model","Method","col_to_sep",
                                     "s_est",
                                     "mu_est","s_snv","m_snv",
                                     "fit_95hdi_low","fit_95hdi_high",
                                     "mut_95hdi_low","mut_95hdi_high",
                                     "aic", "dic")) %>%
      separate(col_to_sep, into = c(NA,NA,NA,"n_presim", "presim_rep","pop"), 
               sep="([_.])") %>%
      mutate(pop = str_replace(pop, "csv", "gln0")) %>%
      mutate(s_est_10 = s_est,
             mu_est_10 = mu_est,
             s_est = 10^s_est,
             mu_est = 10^mu_est) %>%
      mutate(gen = if_else(str_detect(x, "116"), 116, 270))
  }
  if(str_detect(x, "pyABC")==TRUE) {
    file = read_csv(x, col_names = c("Model","Method","col_to_sep",
                                     "s_est",
                                     "mu_est","s_snv","m_snv",
                                     "fit_95hdi_low","fit_95hdi_high",
                                     "mut_95hdi_low","mut_95hdi_high",
                                     "aic", "dic")) %>%
      mutate(n_presim = str_sub(col_to_sep, 1,-2)) %>%
      mutate(pop = paste0("gln0",str_sub(col_to_sep, -1))) %>%
      mutate(presim_rep = case_when(str_detect(x, "_1_") ~ "1",
                                    str_detect(x, "_2_") ~ "2",
                                    str_detect(x, "_3_") ~ "3")) %>%
      select("Model","Method","n_presim", "presim_rep", "pop",
             "s_est",
             "mu_est","s_snv","m_snv",
             "fit_95hdi_low","fit_95hdi_high",
             "mut_95hdi_low","mut_95hdi_high",
             "aic", "dic") %>%
      mutate(s_est_10 = s_est,
             mu_est_10 = mu_est,
             s_est = 10^s_est, 
             mu_est = 10^mu_est)  %>%
        mutate(gen = if_else(str_detect(x, "116"), 116, 270))
  }
  return(file)
}

data_list = map(files,read_est_params)

data = do.call(rbind, data_list)

data_long = data %>% 
  pivot_longer(cols = c(mu_est,s_est), names_to = "parameter",
               values_to = "value")

a = data %>% 
  filter(Model == "WF") %>%
  filter(Method=="SNPE") %>%
  filter(presim_rep=="1") %>%
  filter(n_presim == "100000") %>%
  ggplot(aes(mu_est_10, s_est, color = pop)) +
  geom_point(size = 3, alpha = 0.5) +
  geom_errorbar(aes(ymin = 10^fit_95hdi_low,ymax = 10^fit_95hdi_high)) + 
  geom_errorbarh(aes(xmin = mut_95hdi_low,xmax = mut_95hdi_high)) +
  ylab("MAP s") +
  xlab("log10(MAP μ)") + 
  ggtitle("WF SNPE")

b = data %>% 
  filter(Model == "Chemo") %>%
  filter(Method=="SNPE") %>%
  filter(presim_rep=="1") %>%
  filter(n_presim == "100000") %>%
  ggplot(aes(mu_est_10, s_est, color = pop)) +
  geom_point(size = 3, alpha = 0.5) +
  geom_errorbar(aes(ymin = 10^fit_95hdi_low,ymax = 10^fit_95hdi_high)) + 
  geom_errorbarh(aes(xmin = mut_95hdi_low,xmax = mut_95hdi_high)) +
  ylab("MAP s") +
  xlab("log10(MAP μ)") + 
  ggtitle("Chemo SNPE")

c = data %>% 
  filter(Model == "WF") %>%
  filter(Method=="ABC-SMC") %>%
  filter(presim_rep=="2") %>%
  filter(n_presim == "100000") %>%
  ggplot(aes(mu_est_10, s_est, color = pop)) +
  geom_point(size = 3, alpha = 0.5) +
  geom_errorbar(aes(ymin = 10^fit_95hdi_low,ymax = 10^fit_95hdi_high)) + 
  geom_errorbarh(aes(xmin = mut_95hdi_low,xmax = mut_95hdi_high)) +
  ylab("MAP s") +
  xlab("log10(MAP μ)") + 
  ggtitle("WF ABC-SMC")

d= ggplot() +
  ggtitle("Chemo ABC-SMC")

(a + b)/ (c + d)

#### comparing 270 to 116 ####
data %>% 
  filter(Method == "ABC-SMC") %>%
  #filter(presim_rep=="1") %>%
  #unite(mod_method, Model, Method, sep = " ") %>%
  ggplot(aes(mu_est_10, s_est, color = pop, shape = presim_rep)) +
  geom_point(size = 3, alpha = 0.5) +
  geom_errorbar(aes(ymin = 10^fit_95hdi_low,ymax = 10^fit_95hdi_high)) + 
  geom_errorbarh(aes(xmin = mut_95hdi_low,xmax = mut_95hdi_high)) +
  facet_grid(n_presim~Model) +
  ylab("MAP s") +
  xlab("MAP μ") + 
  ggtitle("WF")

data %>% 
  filter(Model == "Chemo") %>%
  unite(mod_method, Model, Method, sep = " ") %>%
  ggplot(aes(mu_est_10, s_est, color = pop, shape = presim_rep)) +
  geom_point(size = 3, alpha = 0.5) +
  geom_errorbar(aes(ymin = 10^fit_95hdi_low,ymax = 10^fit_95hdi_high)) + 
  geom_errorbarh(aes(xmin = mut_95hdi_low,xmax = mut_95hdi_high)) +
  #scale_color_manual(values=color_vec, 
  #                  name="",
  #                 breaks=mod_meth) +
  facet_grid(n_presim~gen) +
  ylab("MAP s") +
  xlab("MAP μ") + 
  ggtitle("Chemo")

data %>% unite(mod_method, Model, Method, sep = " ") %>%
  ggplot(aes(as.factor(gen), aic, color = n_presim)) +
  geom_boxplot(outlier.color = NA) +
  geom_beeswarm(aes(shape=pop),dodge.width = 0.75, alpha =0.5) +
  ylab("AIC") +
  xlab("")

data %>% unite(mod_method, Model, Method, sep = " ") %>%
  ggplot(aes(as.factor(gen), dic, color = n_presim, shape=pop)) +
  geom_boxplot(outlier.color = NA) +
  geom_beeswarm(dodge.width = 0.75, alpha =0.5) +
  ylab("DIC") +
  xlab("") 

#### OLD BELOW #####
ests = read_csv('Lauer_estimates.csv') %>% select(-delete)


pop_colors=c('#1F77B4', '#FF7F0E', '#2CA02C','#D62728','#9467BD','#8C564B','#E377C2','#7F7F7F','#BCBD22')

ests = ests %>% 
  mutate(s_est = 10^s_est, mu_est=10^mu_est) %>%
  rename(`CNV fitness effect` = s_est, `CNV mutation rate` = mu_est)
  

ests_long = ests %>% 
  pivot_longer(cols = c(`CNV fitness effect`,`CNV mutation rate`), names_to = "parameter",
               values_to = "value")

ests_summary = ests_long %>% group_by(mod, method, parameter) %>%
  summarise(mean_val = mean(value))

mod.labs <- c("Chemostat","Wright-Fisher")
names(mod.labs) <- c("Chemo", "WF")

ggplot(ests_long %>% filter(method == 'pyABC'), 
       aes(parameter, value)) +
  geom_boxplot(outlier.shape = NA) +
  geom_jitter(aes(color = pop), size=4) +
  scale_color_manual(values=pop_colors) +
  theme_light(base_size = 25) +
  labs(color = "Experimental\npopulation") +
  facet_wrap(parameter~mod, scales = "free", labeller =labeller(mod = mod.labs)) +
  ylab("Estimated value") +
  xlab("") +
  ggtitle("pyABC inference on real data")


ggplot(ests %>% filter(method == 'APT') %>% filter(pop != 'gln09'),
       aes(`CNV fitness effect`, `CNV mutation rate`)) +
  geom_smooth(method='lm', se = F, color='black') +
  geom_point(aes(color=pop),size=4) +
  scale_color_manual(values=pop_colors) +
  facet_wrap(~mod, scales = "free", labeller =labeller(mod = mod.labs)) +
  theme_light(base_size = 25) +
  scale_y_continuous(labels = scales::scientific) +
  labs(color = "Experimental\npopulation") 



