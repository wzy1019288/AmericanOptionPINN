library(ggplot2)


p <- ggplot(data=cmb_train,aes(x=dim,y=log(Total.Loss-Free.Boundary.Loss))) +
  stat_boxplot(geom = 'errorbar',width=0.2) +
  geom_boxplot(width=0.2,outlier.shape=1) +
  ylab("Solution Loss - Log Scale") + xlab("Dimensionality") + 
  ggtitle("Solution Losses vs Dimensionality") +
  theme(plot.title = element_text(size = 15, hjust = 0.5),panel.border = element_rect(size = 0.5, colour = "black", fill = NA) )

dat <- ggplot_build(p)$data[[1]]
p <- p + geom_segment(data=dat, aes(x=xmin, xend=xmax, 
                               y=middle, yend=middle), colour="red", size=0.5)


q <- ggplot(data=cmb_train,aes(x=dim,y=log(Free.Boundary.Loss))) +
  stat_boxplot(geom = 'errorbar',width=0.2) +
  geom_boxplot(width=0.2,outlier.shape=1) +
  ylab("Free Boundary Loss - Log Scale") + xlab("Dimensionality") + 
  ggtitle("Free Boundary Losses vs Dimensionality") +
  theme(plot.title = element_text(size = 15, hjust = 0.5),panel.border = element_rect(size = 0.5, colour = "black", fill = NA) )

dat <- ggplot_build(q)$data[[1]]
q <- q + geom_segment(data=dat, aes(x=xmin, xend=xmax, 
                                    y=middle, yend=middle), colour="red", size=0.5)


p5 <- cowplot::plot_grid(p, q, ncol = 2)
p5

