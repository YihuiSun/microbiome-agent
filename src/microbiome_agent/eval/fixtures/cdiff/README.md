# cdiff fixture (case: cdiff-diversity-collapse)

Synthetic, seed=5. 10 healthy + 10 recurrent-CDI samples,
19 features.

Planted signal, modeled on Chang et al., J Infect Dis 2008 (PMID 18199029):
markedly reduced alpha (Shannon) diversity in recurrent C. difficile-associated
diarrhea, via one dominant taxon crowding out the rest of the community.
Verified: mean Shannon in `recurrent_CDI` < 60% of `healthy` mean via
`alpha_diversity` at generation time.
