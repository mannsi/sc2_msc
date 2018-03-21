source .venv/bin/activate

for number in {1..48}
do
python /home/mannsi/Repos/sc2_msc/main_run_agent.py --norender --map DefeatScv --agent my_agents.DiscoverSsmAgent --wait_after_attack "$number" --step_mul 1 --file_log_level 30 --log_file_name steps_between_dmg.txt --max_episodes 1
done