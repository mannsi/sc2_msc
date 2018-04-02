source ../.venv/bin/activate

for number in {1..30}
do
python /home/mannsi/Repos/sc2_msc/main_run_agent.py --norender --map DefeatScv --agent my_agents.AttackAlwaysAgent --step_mul "$number" --file_log_level 30 --log_file_name ssm_results.txt --max_episodes 1
python /home/mannsi/Repos/sc2_msc/main_run_agent.py --norender --map DefeatScv --agent my_agents.AttackMoveAgent --step_mul "$number" --file_log_level 30 --log_file_name ssm_results.txt --max_episodes 1
done