source .venv/bin/activate

for number in {1..40}
do
python /home/mannsi/Repos/sc2_msc/main_run_agent.py --norender --map DefeatScv --agent my_agents.DiscoverSsmAgent --wait_after_attack "$number" --step_mul 1 --file_log_level 20 --log_file_name steps_between_dmg.txt --max_episodes 1
done