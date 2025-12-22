library(bootnet)
library(qgraph)
library(dplyr)
library(ggplot2)

# Load data
data <- read.csv("table_1_2898_label.csv", fileEncoding = "GBK")

hamd_subtype1 <- subset(data, Cluster == 0, select = HAMD1:HAMD17)
hamd_subtype2 <- subset(data, Cluster == 1, select = HAMD1:HAMD17)

# Symptom domain classification
symptom_domain <- c(
  'HAMD1' = 'Affective', 'HAMD2' = 'Affective', 'HAMD3' = 'Affective',
  'HAMD4' = 'Somatic', 'HAMD5' = 'Somatic', 'HAMD6' = 'Somatic',
  'HAMD7' = 'Affective', 'HAMD8' = 'Somatic', 'HAMD9' = 'Affective',
  'HAMD10' = 'Affective', 'HAMD11' = 'Somatic', 'HAMD12' = 'Somatic',
  'HAMD13' = 'Somatic', 'HAMD14' = 'Somatic', 'HAMD15' = 'Affective',
  'HAMD16' = 'Somatic', 'HAMD17' = 'Affective'
)

symptom_labels <- c(
  'HAMD1' = 'Depressed Mood', 'HAMD2' = 'Guilty', 'HAMD3' = 'Suicide',
  'HAMD4' = 'Insomnia Early', 'HAMD5' = 'Insomnia Middle', 'HAMD6' = 'Insomnia Late',
  'HAMD7' = 'Anhedonia', 'HAMD8' = 'Retardation', 'HAMD9' = 'Agitation',
  'HAMD10' = 'Anxiety (Psychic)', 'HAMD11' = 'Anxiety (Somatic)',
  'HAMD12' = 'Gastrointestinal', 'HAMD13' = 'General Somatic',
  'HAMD14' = 'Genital', 'HAMD15' = 'Hypochondriasis',
  'HAMD16' = 'Weight Loss', 'HAMD17' = 'Insight'
)

domain_colors <- c('Affective' = '#7F7FFE', 'Somatic' = '#F98007')

# Node size based on symptom severity
normalize_size <- function(x, min_size = 5, max_size = 20) {
  if (max(x) == min(x)) return(rep(mean(c(min_size, max_size)), length(x)))
  (x - min(x)) / (max(x) - min(x)) * (max_size - min_size) + min_size
}

node_size_subtype1 <- normalize_size(colMeans(hamd_subtype1, na.rm = TRUE))
node_size_subtype2 <- normalize_size(colMeans(hamd_subtype2, na.rm = TRUE))

# Network estimation
network_subtype1 <- estimateNetwork(hamd_subtype1, default = "EBICglasso",
                                    corMethod = "cor", tuning = 0.5)
network_subtype2 <- estimateNetwork(hamd_subtype2, default = "EBICglasso",
                                    corMethod = "cor", tuning = 0.5)

# Network visualization
png("network_comparison.png", width = 4000, height = 2000, res = 400)
par(mfrow = c(1, 2), family = "sans")

plot(network_subtype1, layout = "spring", groups = symptom_domain,
     color = domain_colors, labels = colnames(hamd_subtype1),
     label.scale = FALSE, label.cex = 0.8, label.color = "black",
     vsize = node_size_subtype1, border.width = 1.5, border.color = "gray30",
     edge.color = "#404040", edge.width = 1.2, title = "", legend = FALSE,
     GLratio = 1.8, layout.par = list(repulse.rad = ncol(hamd_subtype1)^2.8))

plot(network_subtype2, layout = "spring", groups = symptom_domain,
     color = domain_colors, labels = colnames(hamd_subtype2),
     label.scale = FALSE, label.cex = 0.8, label.color = "black",
     vsize = node_size_subtype2, border.width = 1.5, border.color = "gray30",
     edge.color = "#404040", edge.width = 1.2, title = "", legend = FALSE,
     GLratio = 1.8, layout.par = list(repulse.rad = ncol(hamd_subtype2)^2.8))

dev.off()

# Centrality analysis
cent_subtype1 <- centrality(network_subtype1)
cent_subtype2 <- centrality(network_subtype2)

centrality_df <- data.frame(
  Symptom = names(symptom_domain),
  Domain = symptom_domain,
  Betweenness_Subtype1 = cent_subtype1$Betweenness,
  Betweenness_Subtype2 = cent_subtype2$Betweenness
)

# Betweenness comparison plot
theme_publication <- theme_classic(base_size = 14, base_family = "sans") +
  theme(
    panel.border = element_rect(color = "black", fill = NA, linewidth = 0.8),
    axis.line = element_blank(),
    axis.ticks = element_line(color = "black", linewidth = 0.8),
    axis.text = element_text(color = "black", size = 12, face = "bold"),
    axis.title = element_text(color = "black", size = 14, face = "bold"),
    axis.text.x = element_text(angle = 45, hjust = 1, size = 11, face = "bold"),
    plot.title = element_text(size = 16, face = "bold", hjust = 0.5),
    legend.position = "bottom",
    legend.title = element_blank(),
    legend.text = element_text(size = 12),
    legend.key.size = unit(1, "lines"),
    legend.background = element_blank(),
    panel.background = element_rect(fill = "white", color = NA),
    plot.background = element_rect(fill = "white", color = NA),
    panel.grid.major = element_line(color = "gray90", linewidth = 0.3),
    panel.grid.minor = element_blank(),
    plot.margin = margin(10, 10, 10, 10)
  )

p_betweenness <- ggplot(centrality_df, aes(x = reorder(Symptom, -Betweenness_Subtype1))) +
  geom_segment(aes(xend = Symptom, y = Betweenness_Subtype1, yend = Betweenness_Subtype2),
               color = "gray60", linewidth = 0.8) +
  geom_point(aes(y = Betweenness_Subtype1, fill = "Subtype 1"),
             size = 4, shape = 21, color = "black", stroke = 0.8) +
  geom_point(aes(y = Betweenness_Subtype2, fill = "Subtype 2"),
             size = 4, shape = 24, color = "black", stroke = 0.8) +
  scale_fill_manual(values = c("Subtype 1" = "#F98007", "Subtype 2" = "#7F7FFE")) +
  scale_x_discrete(labels = symptom_labels) +
  labs(y = "Betweenness Centrality", x = "Symptoms") +
  theme_publication +
  coord_cartesian(clip = "off")

ggsave("betweenness_comparison.png", p_betweenness,
       width = 12, height = 6, dpi = 400, bg = "white")
