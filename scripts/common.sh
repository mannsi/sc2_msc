is_instance_running(){
  server_instance_name="$1"

  status=$(gcloud compute instances describe "$server_instance_name" --format="json" | python -c "import json,sys;obj=json.load(sys.stdin); print obj['status'];")

  if [ "$status" = "TERMINATED" ]; then
    echo "not_running"
  else
    echo "running"
  fi
}


wait_for_ssh_to_be_ready(){
  # Run simple command on server to make sure ssh is ready
  server_instance_name="$1"
  echo "Try ssh until it works"
  until gcloud compute ssh gudbjorn.einarsson@"$server_instance_name" --command "ls > /dev/null"; do
    sleep 5s
  done
  echo "ssh ready"
}

start_instance(){
  server_instance_name="$1"
  gcloud compute instances start "$server_instance_name"
}


get_instance_ip(){
  server_instance_name="$1"

  ip=$(gcloud compute instances describe "$server_instance_name" --format="json" | python -c "import json,sys;obj=json.load(sys.stdin); print obj['networkInterfaces'][0]['accessConfigs'][0]['natIP'];")

  echo "$ip"
}
