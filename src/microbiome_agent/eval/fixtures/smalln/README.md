# smalln fixture (case: permanova-small-n)

Synthetic, seed=8. Only 3 + 3 samples (deliberately below the `run_beta_diversity`
small-group threshold of 4), 20 features, with a clear
compositional separation planted (Fusobacterium/Bacteroides shift, same style
as the CRC signal) so the community really is well-separated. The point of
this case is the PERMANOVA small-n resolution-limit caveat: the MCP server's
`run_beta_diversity` tool emits a literal `note` field whenever a group has
fewer than 4 samples, and the agent is expected to surface that caveat to the
user rather than reporting the p-value at face value.
