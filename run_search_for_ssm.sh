source .venv/bin/activate

# Benchmark without moving
python /home/mannsi/Repos/sc2_msc/main_run_agent.py --norender --map DefeatScv --agent my_agents.DiscoverSsmAgent --wait_after_attack 0 --move_steps_after_attack 0 --step_mul 1 --file_log_level 30 --log_file_name steps_between_dmg.txt --max_episodes 1

for wait_steps in {1..10}
do
    for move_steps_after_attack in {1..15}
    do
    python /home/mannsi/Repos/sc2_msc/main_run_agent.py --norender --map DefeatScv --agent my_agents.DiscoverSsmAgent --wait_after_attack "$wait_steps" --move_steps_after_attack "$move_steps_after_attack" --step_mul 1 --file_log_level 30 --log_file_name steps_between_dmg.txt --max_episodes 1
    done
done