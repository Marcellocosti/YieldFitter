############## CONFIG FOR PP REFERENCE
final_states = {
    "DplusToPiKPi": {
        "query": "abs(fFlagMcMatchRec) == 1",
        "br_sim": 5.00e-1,
        "br_pdg": 9.38e-2
    },
    "DplusToPiKPiPi0": {
      "query": "abs(fFlagMcMatchRec) == 2",
      "br_sim": 3.00e-2,
      "br_pdg": 6.25e-2,
    },
    "DplusToPiPiPi": {
      "query": "abs(fFlagMcMatchRec) == 3",
      "br_sim": 1.1e-1,
      "br_pdg": 3.3e-3,
    },
    "DplusToPiKK": {
      "query": "abs(fFlagMcMatchRec) == 4",
      "br_sim": 3.60e-1,
      "br_pdg": 9.68e-3,
    },
    "DsToPiKK": {
      "query": "abs(fFlagMcMatchRec) == 5",
      "br_sim": 6.50e-1,
      "br_pdg": 5.45e-2,
      "abundance_to_Dplus": 1.25,
    },
    "DsToPiKKPi0": {
      "query": "abs(fFlagMcMatchRec) == 6",
      "br_sim": 2.00e-2,
      "br_pdg": 5.53e-2,
      "abundance_to_Dplus": 1.25,
    },
    "DsToPiPiK": {
      "query": "abs(fFlagMcMatchRec) == 7",
      "br_sim": 1.10e-1,
      "br_pdg": 6.23e-3,
      "abundance_to_Dplus": 1.25,
    },
    "DsToPiPiPi": {
      "query": "abs(fFlagMcMatchRec) == 8",
      "br_sim": 1.10e-1,
      "br_pdg": 1.09e-3,
      "abundance_to_Dplus": 1.25,
    },
    "DsToPiPiPiPi0": {
      "query": "abs(fFlagMcMatchRec) == 9",
      "br_sim": 1.10e-1,
      "br_pdg": 3.8e-2,
      "abundance_to_Dplus": 1.25,
    },
    "DstarD0ToPiKPi": {
      "query": "abs(fFlagMcMatchRec) == 10",
      "br_sim": 5.00e-1,
      "br_pdg": 3.945e-2,
    },
    "DstarD0ToPiKPiPi0": {
      "query": "abs(fFlagMcMatchRec) == 11",
      "br_sim": 1.25e-1,
      "br_pdg": 1.44e-1,
    },
    "DstarD0ToPiKK": {
      "query": "abs(fFlagMcMatchRec) == 13",
      "br_sim": 1.25e-1,
      "br_pdg": 4.08e-3,
    },
    "DstarD0ToPiPiPi": {
      "query": "abs(fFlagMcMatchRec) == 15",
      "br_sim": 1.25e-1,
      "br_pdg": 1.453e-3,
    },
    "DstarD0ToPiPiPiPi0": {
      "query": "abs(fFlagMcMatchRec) == 16",
      "br_sim": 1.25e-1,
      "br_pdg": 1.49e-2,
    }
}
    # "LcToPKPi": {
    #   "query": "abs(fFlagMcMatchRec) == 17",
    #   "br_sim":
    #   "br_pdg":
    # },
    # "LcToPKPiPi0": {
    #   "query": "abs(fFlagMcMatchRec) == 18",
    #   "br_sim":
    #   "br_pdg":
    # },
    # "LcToPPiPi": {
    #   "query": "abs(fFlagMcMatchRec) == 19",
    #   "br_sim":
    #   "br_pdg":
    # },
    # "LcToPKK": {
    #   "query": "abs(fFlagMcMatchRec) == 20",
    #   "br_sim":
    #   "br_pdg":
    # },
    # "XicToPKPi": {
    #   "query": "abs(fFlagMcMatchRec) == 21",
    #   "br_sim":
    #   "br_pdg":
    # },
    # "XicToPKK": {
    #   "query": "abs(fFlagMcMatchRec) == 22",
    #   "br_sim":
    #   "br_pdg":
    # },
    # "XicToSPiPi": {
    #   "query"":"abs(fFlagMcMatchRec) ==  23",
    #   "br_mc"':
    #   "br_pdg"':
    # },































############## CONFIG FOR PBPB
# final_states = {
#     "DplusToPiKPi": {
#         "query": "abs(fFlagMcMatchRec) == 1",
#         "br_sim": 0.28,
#         "br_pdg": 0.0938
#     },
#     "DplusToPiKPiPi0": {
#       "query": "abs(fFlagMcMatchRec) == 2",
#       "br_sim": 5.0e-2,
#       "br_pdg": 6.25e-2,
#     },
#     "DplusToPiPiPi": {
#       "query": "abs(fFlagMcMatchRec) == 3",
#       "br_sim": 3.4e-1,
#       "br_pdg": 3.3e-3,
#     },
#     "DplusToPiKK": {
#       "query": "abs(fFlagMcMatchRec) == 4",
#       "br_sim": 3.3e-1,
#       "br_pdg": 9.68e-3,
#     },
#     "DsToPiKK": {
#       "query": "abs(fFlagMcMatchRec) == 5",
#       "br_sim": 2.0e-1,
#       "br_pdg": 5.45e-2,
#       "abundance_to_Dplus": 1.25,
#     },
#     "DsToPiKKPi0": {
#       "query": "abs(fFlagMcMatchRec) == 6",
#       "br_sim": 5.0e-2,
#       "br_pdg": 5.53e-2,
#       "abundance_to_Dplus": 1.25,
#     },
#     "DsToPiPiK": {
#       "query": "abs(fFlagMcMatchRec) == 7",
#       "br_sim": 2.5e-1,
#       "br_pdg": 6.23e-3,
#       "abundance_to_Dplus": 1.25,
#     },
#     "DsToPiPiPi": {
#       "query": "abs(fFlagMcMatchRec) == 8",
#       "br_sim": 2.5e-1,
#       "br_pdg": 1.09e-3,
#       "abundance_to_Dplus": 1.25,
#     },
#     "DsToPiPiPiPi0": {
#       "query": "abs(fFlagMcMatchRec) == 9",
#       "br_sim": 2.5e-1,
#       "br_pdg": 3.8e-2,
#       "abundance_to_Dplus": 1.25,
#     },
#     "DstarD0ToPiKPi": {
#       "query": "abs(fFlagMcMatchRec) == 10",
#       "br_sim": 2.0e-1,
#       "br_pdg": 3.945e-2,
#     },
#     "DstarD0ToPiKPiPi0": {
#       "query": "abs(fFlagMcMatchRec) == 11",
#       "br_sim": 2.0e-1,
#       "br_pdg": 1.44e-1,
#     },
#     "DstarD0ToPiKK": {
#       "query": "abs(fFlagMcMatchRec) == 13",
#       "br_sim": 2.0e-1,
#       "br_pdg": 4.08e-3,
#     },
#     "DstarD0ToPiPiPi": {
#       "query": "abs(fFlagMcMatchRec) == 15",
#       "br_sim": 2.0e-1,
#       "br_pdg": 1.453e-3,
#     },
#     "DstarD0ToPiPiPiPi0": {
#       "query": "abs(fFlagMcMatchRec) == 16",
#       "br_sim": 2.0e-1,
#       "br_pdg": 1.49e-2,
#     }
# }
#     # "LcToPKPi": {
#     #   "query": "abs(fFlagMcMatchRec) == 17",
#     #   "br_sim":
#     #   "br_pdg":
#     # },
#     # "LcToPKPiPi0": {
#     #   "query": "abs(fFlagMcMatchRec) == 18",
#     #   "br_sim":
#     #   "br_pdg":
#     # },
#     # "LcToPPiPi": {
#     #   "query": "abs(fFlagMcMatchRec) == 19",
#     #   "br_sim":
#     #   "br_pdg":
#     # },
#     # "LcToPKK": {
#     #   "query": "abs(fFlagMcMatchRec) == 20",
#     #   "br_sim":
#     #   "br_pdg":
#     # },
#     # "XicToPKPi": {
#     #   "query": "abs(fFlagMcMatchRec) == 21",
#     #   "br_sim":
#     #   "br_pdg":
#     # },
#     # "XicToPKK": {
#     #   "query": "abs(fFlagMcMatchRec) == 22",
#     #   "br_sim":
#     #   "br_pdg":
#     # },
#     # "XicToSPiPi": {
#     #   "query"":"abs(fFlagMcMatchRec) ==  23",
#     #   "br_mc"':
#     #   "br_pdg"':
#     # },


# ####### ALL CHANNELS SEPARATED
# # # Each channel must be reweighted by the ratio of the BRs in MC and PDG
# # final_states_dplus = {
# #     "DplusToPiKPi": {
# #         "FlagFinal": 1,
# #         "ResoStates": [
# #             {
# #                 "Channel": "NonResonant",
# #                 "FlagReso": 0,
# #                 'br_mc': 0.28,
# #                 'br_pdg': 0.0938
# #             }
# #         ]
# #     },
# #     "DplusToPiKPiPi0": {
# #         "FlagFinal": 2,
# #         "ResoStates": [
# #             {
# #                 "Channel": "NonResonant",
# #                 "FlagReso": 0,
# #                 'br_mc': 0.05,
# #                 'br_pdg': 0.0625
# #             }
# #         ]
# #     },
# #     "DplusToPiKK": {
# #         "FlagFinal": 4,
# #         "ResoStates": [
# #             {
# #                 "Channel": "K0892_KPlus",
# #                 "FlagReso": 2,
# #                 'br_mc': 0.085,
# #                 'br_pdg': 2.49e-3
# #             },
# #             {
# #                 "Channel": "K1430_KPlus",
# #                 "FlagReso": 3,
# #                 'br_mc': 0.062,
# #                 'br_pdg': 1.82e-3
# #             },
# #             {
# #                 "Channel": "Phi_Pi",
# #                 "FlagReso": 1,
# #                 'br_mc': 0.092,
# #                 'br_pdg': 2.69e-3
# #             },
# #             {
# #                 "Channel": "NonResonant",
# #                 "FlagReso": 0,
# #                 'br_mc': 0.091,
# #                 'br_pdg': 2.68e-3 ## How to assign PDG BRs to non-resonant channels?
# #             }
# #         ]
# #     },
# #     "DplusToPiPiPi": {
# #         "FlagFinal": 3,
# #         "ResoStates": [
# #             {
# #                 "Channel": "Rho0Pi",
# #                 "FlagReso": 4,
# #                 'br_mc': 0.09,
# #                 'br_pdg': 8.4e-4
# #             },
# #             {
# #                 "Channel": "f21270_Pi",
# #                 "FlagReso": 5,
# #                 'br_mc': 0.05,
# #                 'br_pdg': 4.6e-4
# #             },
# #             {
# #                 "Channel": "NonResonant",
# #                 "FlagReso": 0,
# #                 'br_mc': 0.20,
# #                 'br_pdg': 1.0e-4
# #             }
# #         ]
# #     }
# # }

# # final_states_ds = {
# #     "DsToPiKK": {
# #         "FlagFinal": 5,
# #         "ResoStates": [
# #             {
# #                 "Channel": "Phi_Pi",
# #                 "FlagReso": 6,
# #                 'br_mc': 0.1,
# #                 'br_pdg': 0.0225
# #             },
# #             {
# #                 "Channel": "K892_K",
# #                 "FlagReso": 8,
# #                 'br_mc': 0.1,
# #                 'br_pdg': 0.0261
# #             }
# #         ]
# #     },
# #     "DsToPiKKPi0": {
# #         "FlagFinal": 6,
# #         "ResoStates": [
# #             {
# #                 "Channel": "Phi_Rho",
# #                 "FlagReso": 7,
# #                 'br_mc': 0.05,
# #                 'br_pdg': 0.059
# #             }
# #         ]
# #     },
# #     "DsToPiPiPi": {
# #         "FlagFinal": 8,
# #         "ResoStates": [
# #             {
# #                 "Channel": "Rho_Pi",
# #                 "FlagReso": 10,
# #                 'br_mc': 0.005,
# #                 'br_pdg': 1.14e-4
# #             },
# #             {
# #                 "Channel": "f21270_Pi",
# #                 "FlagReso": 12,
# #                 'br_mc': 0.005,
# #                 'br_pdg': 1.42e-3
# #             },
# #             {
# #                 "Channel": "NonResonant",
# #                 "FlagReso": 0,
# #                 'br_mc': 0.24,
# #                 'br_pdg': 9.23e-3
# #             }
# #         ]
# #     },
# #     "DsToPiPiK": {
# #         "FlagFinal": 7,
# #         "ResoStates": [
# #             {
# #                 "Channel": "K0892_Pi",
# #                 "FlagReso": 9,
# #                 'br_mc': 0.07,
# #                 'br_pdg': 1.68e-3
# #             },
# #             {
# #                 "Channel": "f01370_K",
# #                 "FlagReso": 13,
# #                 'br_mc': 0.05,
# #                 'br_pdg': 1.2e-3
# #             },
# #             {
# #                 "Channel": "Rho0_K",
# #                 "FlagReso": 11,
# #                 'br_mc': 0.09,
# #                 'br_pdg': 2.18e-3
# #             },
# #             {
# #                 "Channel": "NonResonant",
# #                 "FlagReso": 0,
# #                 'br_mc': 0.04,
# #                 'br_pdg': 9.9e-4
# #             }
# #         ]
# #     },
# #     "DsToPiPiPiPi0": {
# #         "FlagFinal": 9,
# #         "ResoStates": [
# #             {
# #                 "Channel": "Eta_Pi",
# #                 "FlagReso": 14,
# #                 'br_mc': 0.25,
# #                 'br_pdg': 0.01686
# #             }
# #         ]
# #     }
# # }

# # # D* TO D0 BRANCHING RATIO IS EQUAL IN PDG AND MC --> REPORT BRS FOR D0
# # final_states_dstar_to_d0_piplus = {
# #     "DstarD0ToPiKPi": {     # Fully reconstructed in D+ inv. mass. spectrum
# #         "FlagFinal": 10,
# #         "ResoStates": [
# #             {
# #                 "Channel": "NonResonant",
# #                 "FlagReso": 0,
# #                 'br_mc': 0.20,
# #                 'br_pdg': 0.03945
# #             }
# #         ]
# #     },
# #     "DstarD0ToPiKPiPi0": {
# #         "FlagFinal": 11,
# #         "ResoStates": [
# #             {
# #                 "Channel": "NonResonant",
# #                 "FlagReso": 0,
# #                 'br_mc': 0.01,
# #                 'br_pdg': 0.0115
# #             },
# #             {
# #                 "Channel": "Rho_K",
# #                 "FlagReso": 16,
# #                 'br_mc': 0.14,
# #                 'br_pdg': 0.112
# #             },
# #             {
# #                 "Channel": "K0892_Pi0",
# #                 "FlagReso": 17,
# #                 'br_mc': 0.02,
# #                 'br_pdg': 0.0195
# #             },
# #             {
# #                 "Channel": "KMinus892_Pi",
# #                 "FlagReso": 18,
# #                 'br_mc': 0.03,
# #                 'br_pdg': 0.0231
# #             }
# #         ]
# #     },
# #     "DstarD0ToPiKK": {
# #         "FlagFinal": 13,
# #         "ResoStates": [
# #             {
# #                 "Channel": "NonResonant",
# #                 "FlagReso": 0,
# #                 'br_mc': 0.20,
# #                 'br_pdg': 4.08e-3
# #             }
# #         ]
# #     },
# #     "DstarD0ToPiPiPi": {
# #         "FlagFinal": 15,
# #         "ResoStates": [
# #             {
# #                 "Channel": "NonResonant",
# #                 "FlagReso": 0,
# #                 'br_mc': 0.20,
# #                 'br_pdg': 1.453e-3
# #             }
# #         ]
# #     },
# #     "DstarD0ToPiPiPiPi0": {
# #         "FlagFinal": 16,
# #         "ResoStates": [
# #             {
# #                 "Channel": "NonResonant",
# #                 "FlagReso": 0,
# #                 'br_mc': 0.06,
# #                 'br_pdg': 1.3e-4
# #             },
# #             {
# #                 "Channel": "Rho_Pi",
# #                 "FlagReso": 15,
# #                 'br_mc': 0.14,
# #                 'br_pdg': 0.0101
# #             }
# #         ]
# #     }
# # }

# # # D* TO D+ BRANCHING RATIO IS EQUAL IN PDG AND MC --> REPORT BRS FOR D+
# # final_states_dstar_to_dplus_pi0 = {
# #     "DstarDplusToPiKPiPi0": {
# #         "FlagFinal": 11,
# #         "ResoStates": [
# #             {
# #                 "Channel": "NonResonant",
# #                 "FlagReso": 0,
# #                 'br_mc': 0.28,
# #                 'br_pdg': 0.0938
# #             }
# #         ]
# #     },
# #     "DstarDplusToPiKPiPi0Pi0": {
# #         "FlagFinal": 12,
# #         "ResoStates": [
# #             {
# #                 "Channel": "NonResonant",
# #                 "FlagReso": 0,
# #                 'br_mc': 0.05,
# #                 'br_pdg': 0.0625
# #             }
# #         ]
# #     },
# #     "DstarDplusToPiKKPi0": {
# #         "FlagFinal": 14,
# #         "ResoStates": [
# #             {
# #                 "Channel": "K0892_KPlus",
# #                 "FlagReso": 20,
# #                 'br_mc': 0.085,
# #                 'br_pdg': 2.49e-3
# #             },
# #             {
# #                 "Channel": "K1430_KPlus",
# #                 "FlagReso": 21,
# #                 'br_mc': 0.062,
# #                 'br_pdg': 1.82e-3
# #             },
# #             {
# #                 "Channel": "Phi_Pi",
# #                 "FlagReso": 19,
# #                 'br_mc': 0.092,
# #                 'br_pdg': 2.69e-3
# #             },
# #             {
# #                 "Channel": "NonResonant",
# #                 "FlagReso": 0,
# #                 'br_mc': 0.091,
# #                 'br_pdg': 2.68e-3 ## How to assign PDG BRs to non-resonant channels?
# #             }
# #         ]
# #     },
# #     "DstarDplusToPiPiPiPi0": {
# #         "FlagFinal": 16,
# #         "ResoStates": [
# #             {
# #                 "Channel": "Rho0Pi",
# #                 "FlagReso": 4,
# #                 'br_mc': 0.09,
# #                 'br_pdg': 8.4e-4
# #             },
# #             {
# #                 "Channel": "f21270_Pi",
# #                 "FlagReso": 5,
# #                 'br_mc': 0.05,
# #                 'br_pdg': 4.6e-4
# #             },
# #             {
# #                 "Channel": "NonResonant",
# #                 "FlagReso": 0,
# #                 'br_mc': 0.20,
# #                 'br_pdg': 1.0e-4
# #             }
# #         ]
# #     }
# # }

# # final_states_lc = {
# #     "LcToPKPi": {
# #         "FlagFinal": 17,
# #         "ResoStates": [
# #             {
# #                 "Channel": "NonResonant",
# #                 "FlagReso": 0,
# #                 'br_mc': 0.08,
# #                 'br_pdg': 0.035
# #             }, 
# #             {
# #                 "Channel": "P_K0892",
# #                 "FlagReso": 24,
# #                 'br_mc': 0.045,
# #                 'br_pdg': 0.0141
# #             },
# #             {
# #                 "Channel": "DeltaPlusPlus_K",
# #                 "FlagReso": 25,
# #                 'br_mc': 0.025,
# #                 'br_pdg': 0.0179
# #             },
# #             {
# #                 "Channel": "L1520_Pi",
# #                 "FlagReso": 26,
# #                 'br_mc': 0.05,
# #                 'br_pdg': 1.18e-3
# #             }
# #         ]
# #     },
# #     "LcToPKPiPi0": {
# #         "FlagFinal": 18,
# #         "ResoStates": [
# #             {
# #                 "Channel": "NonResonant",
# #                 "FlagReso": 0,
# #                 'br_mc': 0.05,
# #                 'br_pdg': 0.0452
# #             }
# #         ]
# #     },
# #     "LcToPPiPi": {
# #         "FlagFinal": 19,
# #         "ResoStates": [
# #             {
# #                 "Channel": "NonResonant",
# #                 "FlagReso": 0,
# #                 'br_mc': 0.25,
# #                 'br_pdg': 4.67e-3
# #             }
# #         ]
# #     },
# #     "LcToPKK": {
# #         "FlagFinal": 20,
# #         "ResoStates": [
# #             {
# #                 "Channel": "P_Phi",
# #                 "FlagReso": 0,     # NOT PRESENT IN O2PHYSICS BY MISTAKE
# #                 'br_mc': 0.25,
# #                 'br_pdg': 0.5e-3
# #             },
# #         ]
# #     }
# # }

# # final_states_xic = {
# #     "XicToPKPi": {
# #         "FlagFinal": 21,
# #         "ResoStates": [
# #             {
# #                 "Channel": " NonResonant",
# #                 "FlagReso": 0,
# #                 'br_mc': 0.17,
# #                 'br_pdg': 2.9e-3
# #             },
# #             {
# #                 "Channel": "P_K0892",
# #                 "FlagReso": 27,
# #                 'br_mc': 0.17,
# #                 'br_pdg': 3.3e-3
# #             }
# #         ]
# #     },
# #     "XicToPKK": {
# #         "FlagFinal": 22,
# #         "ResoStates": [
# #             {
# #                 "Channel": "P_Phi",
# #                 "FlagReso": 28,
# #                 'br_mc': 0.33,
# #                 'br_pdg': 0.6e-4
# #             }
# #         ]
# #     },
# #     "XicToSPiPi": {
# #         "FlagFinal": 23,
# #         "ResoStates": [
# #             {
# #                 "Channel": "NonResonant",
# #                 "FlagReso": 0,
# #                 'br_mc': 0.33,
# #                 'br_pdg': 0.014
# #             }
# #         ]
# #     }
# # }
