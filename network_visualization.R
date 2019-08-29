library('igraph')

network = graph_from_literal(
'Input' - '1st Conv1D (2x1x2)' - '2nd Conv1D (2x1x2)' - '3rd Conv1D (2x1x2)' - 'Flatten (2x1x2)' - 'Dense (2x1x2)' - 'Dropout (2x1x2)',
'Input' - '1st Conv1D (2x1x3)' - '2nd Conv1D (2x1x3)' - '3rd Conv1D (2x1x3)' - 'Flatten (2x1x3)' - 'Dense (2x1x3)' - 'Dropout (2x1x3)',
'Input' - '1st Conv1D (3x1x2)' - '2nd Conv1D (3x1x2)' - '3rd Conv1D (3x1x2)' - 'Flatten (3x1x2)' - 'Dense (3x1x2)' - 'Dropout (3x1x2)',
'Input' - '1st Conv1D (3x1x3)' - '2nd Conv1D (3x1x3)' - '3rd Conv1D (3x1x3)' - 'Flatten (3x1x3)' - 'Dense (3x1x3)' - 'Dropout (3x1x3)',
'Dropout (2x1x2)' - '1st Dense (Global)',
'Dropout (2x1x3)' - '1st Dense (Global)',
'Dropout (3x1x2)' - '1st Dense (Global)',
'Dropout (3x1x3)' - '1st Dense (Global)',
'1st Dense (Global)' - 'Dropout (Global)' - 'Repeat Vector' - '1st GRU' - '2nd GRU' - '3rd GRU' - 'Time-Distributed Dense'
)

graph(network)