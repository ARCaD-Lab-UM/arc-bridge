Using LCM tools to record (log) or play LCM messages.

Make the scripts executable:
    chmod +x record_lcm.sh play_lcm.sh

Record (log) all LCM messages in the current directory:
    ./record_lcm.sh
This runs "lcm-logger" and keeps recording until one press Ctrl-C.

Play back a recorded LCM log:
    ./play_lcm.sh <path-to-lcm-log>
This runs "lcm-logplayer-gui" on the provided log file and keeps playing 
until one press Ctrl-C (or close the GUI).

Notes:
Depending on the modules, one may need to change the Playback Channel name(s) in the GUI.
This folder's .gitignore ignores everything except:
    - readme.txt
    - record_lcm.sh
    - play_lcm.sh
