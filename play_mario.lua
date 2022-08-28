FRAMES_PER_SCREENSHOT = 10
SAVE_STATE_FILE = "savestates/CustomState1.state"

-- VARIOUS RAM LOCATIONS USED IN THIS FILE
BYTE_RUN_GAME = 0x10				--1 BYTE
BYTE_EFFECTIVE_FRAME_COUNTER = 0x14	--1 BYTE
BYTE_GAME_MODE = 0xD9B				--1 BYTE
BYTE_LEVEL_ID = 0x17BB				--1 BYTE

BYTE_MARIO_STATE = 0x71			--1 BYTE	possible values: https://www.smwcentral.net/?p=memorymap&game=smw&region=ram&address=7E0071&context=

BYTE_MARIO_X = 0x94
BYTE_MARIO_Y = 0x96
BYTE_LAYER1X = 0x1A
BYTE_LAYER1Y = 0x1C

function getMarioState()
	local marioX = mainmemory.read_s16_le(BYTE_MARIO_X)
	local marioY = mainmemory.read_s16_le(BYTE_MARIO_Y)
	local marioState = mainmemory.read_s8(BYTE_MARIO_STATE)

	return marioX, marioY, marioState
end

ButtonNames = {
	"A",
	"B",
	"X",
	"Y",
	"Up",
	"Down",
	"Left",
	"Right",
}

function getController()
	local buttons = {}
	local state = joypad.get(1)
	
	for b=1,#ButtonNames do
		button = ButtonNames[b] 
		if state[button] then
			buttons[b] = 1
		else
			buttons[b] = 0
		end
	end
	
	return buttons
end

function writeStateToFile(id, controller_state, ...)
	local stateFile = io.open(STATE_FILE_NAME, "a")

	stateFile:write(id)
	for i, value in ipairs(controller_state) do
		stateFile:write("," .. value)
	end
	for i, value in ipairs(arg) do
		stateFile:write("," .. value)
	end
	stateFile:write("\n")

	io.close(stateFile)
end

function readButtonsFromFile()
	local stateFile = io.open(BUTTONS_FILE_NAME, "r")
end

function clearJoypad()
	controller = {}
	for b = 1,#ButtonNames do
		controller["P1 " .. ButtonNames[b]] = false
	end
	joypad.set(controller)
end

function setup()
	--Checks for idling
	max_mario_x = 0
	idle_time = 0
	alive = false

	--New file names for every timestep
	time_stamp = os.time(os.date("!*t"))
	id = 0
	STATE_FILE_NAME = "experiments/experiment_" .. experiment_id .. "/data/data_" .. time_stamp .. "/stateFile.csv"
	BUTTONS_FILE_NAME = "experiments/experiment_" .. experiment_id .. "/data/data_" .. time_stamp .. "/outputFile.csv"

	--Make state file
	os.execute("mkdir experiments")
	os.execute("cd experiments && mkdir experiment_" .. experiment_id)
	os.execute("cd experiments/experiment_" .. experiment_id .. " && mkdir data")
	os.execute("cd experiments/experiment_" .. experiment_id .. "/data && mkdir data_" .. time_stamp)

	local stateFile = io.open(STATE_FILE_NAME, "w")
	stateFile:write("id,A,B,X,Y,Up,Down,Left,Right,MarioX,MarioY,MarioState\n")
	io.close(stateFile)

	--Go back to savestate
	savestate.load(SAVE_STATE_FILE)
	timeout = TimeoutConstant
	clearJoypad()

	--Setup other stuff
	load_level = false
end

function checkInLevel()
	game_mode = mainmemory.read_u8(BYTE_GAME_MODE)
	return game_mode == 0 or game_mode == 1
end

function split(inputstr, sep)
	--Helper function because lua does not have a split function for some reason
	--returns table split based on character as given in sep
	if sep == nil then
			sep = "%s"
	end
	local t={}
	for str in string.gmatch(inputstr, "([^"..sep.."]+)") do
			table.insert(t, str)
	end
	return t
end

function getLastLine(inputstr)
	--Return last line of a file
	t = split(inputstr, "\n")
	return t[#t]
end

function readOutputFile()
	local f = io.open(BUTTONS_FILE_NAME, "r")
	if f == nil then
		return nil
	end
	f:seek("end", -100)

	local text = f:read(100)
	local lastLine = getLastLine(text)

	f:close()

	return lastLine
end

function isempty(s)
	return s == nil or s == ''
end

function setController(line)
	if not isempty(line) and line ~= nil then
		t = split(line, ',')
		local controller = {}

		for i=1, #ButtonNames do 
			local button = "P1 " .. ButtonNames[i]

			controller[button] = t[i + 1] == "1"
		end

		joypad.set(controller)
	end
end

function get_id_from_output_file(lastOutput)
	if not isempty(lastOutput) and lastOutput ~= nil then
		t = split(lastOutput, ',')
		if t[1] == 'id' then --first line
			last_output_id = -1
		else
			last_output_id = t[1]
		end
	else
		last_output_id = -1
	end
	return last_output_id
end


function mainLoop()
	while true do
		if do_run then
			if checkInLevel() then
				lastOutput = readOutputFile()

				if id % FRAMES_PER_SCREENSHOT == 0 then
					client.screenshot("experiments/experiment_" .. experiment_id .. "/data/data_" .. time_stamp .. "/screenshot_" .. id .. ".png")

					last_output_id = get_id_from_output_file(lastOutput)
					while (id - last_output_id > 10) and (not first_run) do
						lastOutput = readOutputFile()
						last_output_id = get_id_from_output_file(lastOutput)
					end
				end
			
				controller_state = getController()
				marioX, marioY, marioState = getMarioState()

				--Update idle time
				if marioX <= max_mario_x then
					idle_time = idle_time + 1
				else
					max_mario_x = marioX
					idle_time = 0
				end

				--update joystick
				setController(lastOutput)

				writeStateToFile(id, controller_state, marioX, marioY, marioState)

				id = id + 1

				--If mario is idle for 15 seconds, restart
				if idle_time > 900 then
					first_run = false
					setup()
				end

				--If mario is dead after being not dead previousily, restart
				local marioState = mainmemory.read_s8(BYTE_MARIO_STATE)
				if alive and marioState == 9 then
					first_run = false
					setup()
				elseif marioState == 0 then
					alive=true
				end
			else
				first_run = false
				setup()
			end
		end
		emu.frameadvance()
	end
end

do_run = false
function startExperiment()
	first_run = true
	experiment_id = tonumber(forms.gettext(form_exp_id))
	if not do_run then
		local continue_string = forms.ischecked(form_continue_from_last_box) and " --continue_from_last" or ""
		
		local obj_speed = forms.ischecked(form_obj_speed_box) and " speed" or ""
		local obj_death = forms.ischecked(form_obj_death_box) and " death" or ""
		obj_string = obj_speed .. obj_death
		if not (obj_string == '') then
			obj_string = " -o " .. obj_string
		end

		local use_weighted_string = forms.ischecked(form_use_weighted_data_box) and " --use_weighted_dataloader" or ""

		setup()
		io.popen("start python.exe watch.py -e " .. experiment_id .. obj_string .. use_weighted_string)
		io.popen("start python.exe train.py --model_path experiments/experiment_" .. experiment_id .. "/models/ --data_path experiments/experiment_" .. experiment_id .. "/data --num_workers 2 --sleep 10" .. continue_string .. obj_string .. use_weighted_string)
		forms.settext(form_start_button, "Stop")
		do_run = true
	else
		os.execute("taskkill /IM python.exe /F")
		forms.settext(form_start_button, "Start")
		do_run = false
	end
end

function makeForm()
	form = forms.newform(500, 500, "nn_mario")

	--Create form from bottom to top, because of z-index rendering issues
	form_start_button = forms.button(form, "Start", startExperiment, 120, 175, 100, 20)
	
	form_use_weighted_data_text = forms.label(form, "Use weighted data", 5, 150)
	form_use_weighted_data_box = forms.checkbox(form, "", 120, 145)

	form_continue_from_last_text = forms.label(form, "Continue last model", 5, 125)
	form_continue_from_last_box = forms.checkbox(form, "", 120, 120)
	
	form_exp_id = forms.textbox(form, "0", 100, 20, "UNSIGNED", 120, 100)
	form_id_text = forms.label(form, "Experiment ID", 5, 100)
	
	--Right side
	form_obj_death_text = forms.label(form, "Mario death", 275, 150)
	form_obj_death_box = forms.checkbox(form, "", 260, 145)
	form_obj_speed_text = forms.label(form, "Mario speed", 275, 125)
	form_obj_speed_box = forms.checkbox(form, "", 260, 120)
	form_obj_text = forms.label(form, "Which training objectives to use:", 260, 100, 200, 25)

	--Top text
	form_welcome_message1 = forms.label(form, "Note: all python.exe tasks will be killed when stopped.", 5, 53, 500, 25, false)
	form_welcome_message1 = forms.label(form, "otherwise continue with previous data.", 5, 38, 500, 25, false)
	form_welcome_message2 = forms.label(form, "If experiment ID is set to a new value, start with zero amount of data and new neural network,", 5, 23, 500, 25, false)
	form_welcome_message3 = forms.label(form, "Form for mario_neural_network to train and play Super Mario World!", 5, 8, 500, 25, false)
end

makeForm()
mainLoop()