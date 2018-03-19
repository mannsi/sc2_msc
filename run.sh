source .venv/bin/activate

for number in {1..30}
do
python /home/mannsi/Repos/sc2_msc/main.py --norender --map DefeatScv --agent my_agents.AttackAlwaysAgent --step_mul "$number" --file_log_level 30 --log_file_name ssm_results.txt
python /home/mannsi/Repos/sc2_msc/main.py --norender --map DefeatScv --agent my_agents.AttackMoveAgent --step_mul "$number" --file_log_level 30 --log_file_name ssm_results.txt
done