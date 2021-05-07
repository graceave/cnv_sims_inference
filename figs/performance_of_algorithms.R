# performance of delfi
# g avecilla
# 8 Apr 2020

library(tidyverse)
library(ggbeeswarm)
library(patchwork)
setwd("/Volumes/GoogleDrive/.shortcut-targets-by-id/19NQRq-XLPpjqJGmLbP4MHRjala2WVahk/Avecilla et al Simulating CNV evolution/scripts/cnv_sims_inference/hpc_output/test_performance/param_estimates/")

data_delfi = read_csv('est_real_params.csv') %>%
  dplyr::filter(mu_snv > 0) 
data_delfi = data_delfi %>%
  mutate(sim_id = rownames(data_delfi))
data_abc = read.csv("est_real_params_abc.csv") %>%
  dplyr::filter(mu_real > 0)


#transform back to actual rates and effects 
data_delfi = data_delfi %>% 
  mutate(s_est = 10^s_est, s_real = 10^s_real,
         mu_est = 10^mu_est, mu_real = 10^mu_real) 

delfi_transformed = data_delfi %>%
  mutate(mu_ratio = mu_est/mu_real, s_ratio = s_est/s_real)

abc_transformed = data_abc %>%
  mutate(mu_ratio = mu_est/mu_real, s_ratio = s_est/s_real)


#CNV mutation rate
# New facet label names for SNV fitness variable
s_snv.labs <- c("SNV fitness 0.001","SNV fitness 0.1")
names(s_snv.labs) <- c("0.001", "0.1")

# New facet label names for SNV mutation rate variable
m_snv.labs <- c("SNV mutation rate 1e-7", "SNV mutation rate 1e-5")
names(m_snv.labs) <- c("1e-07", "1e-05")

# Create the plot
a <- ggplot(delfi_transformed,
       aes(as.character(mu_real), 
           mu_ratio, color=as.character(s_real))) +
  geom_boxplot() +
  #geom_jitter() +
  theme_light(base_size = 12) +
  labs(color = "CNV fitness effect") +
  facet_grid(mu_snv~s_snv, 
             labeller =labeller(mu_snv = m_snv.labs,s_snv = s_snv.labs)) +
  xlab('CNV mutation rate') +
  ylab('log10(CNV estimated/real mutation rate)') + 
  scale_y_log10(breaks = c(0.01, 0.10, 1, 10.00, 100.00), limits = c(0.012, 100)) +
  scale_color_manual(values = c("#8dd3c7", "#bebada")) +
  ggtitle("Inference with SNPE")

b <- ggplot(abc_transformed,
            aes(as.character(mu_real), 
                mu_ratio, color=as.character(s_real))) +
  geom_boxplot() +
  #geom_jitter() +
  theme_light(base_size = 12) +
  labs(color = "CNV fitness effect") +
  facet_grid(mu_snv~s_snv, 
             labeller =labeller(mu_snv = m_snv.labs,s_snv = s_snv.labs)) +
  xlab('CNV mutation rate') +
  ylab('log10(CNV estimated/real mutation rate)') + 
  scale_y_log10(breaks = c(0.01, 0.10, 1, 10.00, 100.00), limits = c(0.012, 100)) +
  scale_color_manual(values = c("#8dd3c7", "#bebada")) +
  ggtitle("Inference with ABC")

# CNV fitness effect
c <- ggplot(delfi_transformed, 
       aes(as.character(mu_real), 
           s_ratio, color=as.character(s_real))) +
  geom_boxplot() +
  #geom_jitter() +
  theme_light(base_size = 12) +
  labs(color = "CNV fitness effect") +
  facet_grid(mu_snv~s_snv, 
             labeller =labeller(mu_snv = m_snv.labs,s_snv = s_snv.labs)) +
  xlab('CNV mutation rate') +
  ylab('log10(CNV estimated/real fitness effect)') + 
  scale_y_log10(breaks = c(0.01, 0.10, 1, 10.00, 100.00), limits = c(0.1, 100)) +
  scale_color_manual(values = c("#8dd3c7", "#bebada")) +
  ggtitle("Inference with SNPE")

d <- ggplot(abc_transformed, 
            aes(as.character(mu_real), 
                s_ratio, color=as.character(s_real))) +
  geom_boxplot() +
  #geom_jitter() +
  theme_light(base_size = 12) +
  labs(color = "CNV fitness effect") +
  facet_grid(mu_snv~s_snv, 
             labeller =labeller(mu_snv = m_snv.labs,s_snv = s_snv.labs)) +
  xlab('CNV mutation rate') +
  ylab('log10(CNV estimated/real fitness effect)') + 
  scale_y_log10(breaks = c(0.01, 0.10, 1, 10.00, 100.00), limits = c(0.1, 200)) +
  scale_color_manual(values = c("#8dd3c7", "#bebada")) +
  ggtitle("Inference with ABC")


(a | b) / (c|d)


## on real data ##
delfi_real = read_csv("realdata_params.csv")
delfi_real = delfi_real %>% 
  rename(`CNV fitness effect` = s_est, `CNV mutation rate` = mu_est) %>%
  pivot_longer(cols = c(`CNV fitness effect`,`CNV mutation rate`), names_to = "parameter",
               values_to = "value") %>%
  mutate(value = 10^value)

ggplot(delfi_real, aes(parameter, value)) +
  geom_boxplot() +
  geom_jitter() +
  theme_light(base_size = 14) +
  labs(color = "CNV fitness effect") +
  facet_wrap(~parameter, scales = "free") +
  ylab("Estimated value") +
  xlab("")
