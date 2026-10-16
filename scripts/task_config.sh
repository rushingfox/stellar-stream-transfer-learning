# Sourced by the data-preparation and submission scripts: maps TASK to the
# experiment-folder prefix and the raw-data file prefix. Mirrors TASKS in
# src/exp_paths.py; keep the two in sync.
#   TASK=prog_mass (default)  <target_columns>_<size>/          raw_data/prog_mass_reg_dataset_<size>.h5
#   TASK=subhalo              subhalo_<target_columns>_<size>/  raw_data/subhalo_reg_dataset_<size>.h5
task=${TASK:-prog_mass}
case "${task}" in
    prog_mass) task_dir_prefix="";         task_raw_prefix="prog_mass_reg_dataset" ;;
    subhalo)   task_dir_prefix="subhalo_"; task_raw_prefix="subhalo_reg_dataset" ;;
    *) echo "Error: unknown TASK='${task}' (expected prog_mass or subhalo)" >&2; exit 1 ;;
esac
