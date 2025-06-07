import MDAnalysis
import MDAnalysisData

adk = MDAnalysisData.datasets.fetch_adk_equilibrium(data_home='protein')
adk_data = MDAnalysis.Universe(adk.topology, adk.trajectory)