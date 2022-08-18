FRAMES_PER_SCREENSHOT = 10
SAVE_STATE_FILE = "savestates/CustomState1.state"

-- VARIOUS RAM LOCATIONS USED IN THIS FILE
BYTE_RUN_GAME = 0x10				--1 BYTE
BYTE_EFFECTIVE_FRAME_COUNTER = 0x14	--1 BYTE
BYTE_GAME_MODE = 0xD9B				--1 BYTE
BYTE_LEVEL_ID = 0x17BB				--1 BYTE

BYTE_MARIO_X = 0x94
BYTE_MARIO_Y = 0x96
BYTE_LAYER1X = 0x1A
BYTE_LAYER1Y = 0x1C

function getPositions()
	local marioX = mainmemory.read_s16_le(BYTE_MARIO_X)
	local marioY = mainmemory.read_s16_le(BYTE_MARIO_Y)

	return marioX, marioY
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
	prev_mario_x = nil
	prev_mario_y = nil
	idle_time = 0

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
	stateFile:write("id,A,B,X,Y,Up,Down,Left,Right,MarioX,MarioY,LevelID\n")
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

function mainLoop()
	while true do
		if do_run then
			if checkInLevel() then
				if id % FRAMES_PER_SCREENSHOT == 0 then
					client.screenshot("experiments/experiment_" .. experiment_id .. "/data/data_" .. time_stamp .. "/screenshot_" .. id .. ".png")
				end
			
				controller_state = getController()
				marioX, marioY = getPositions()

				--Update idle time
				if marioX == prev_mario_x and marioY == prev_mario_y then
					idle_time = idle_time + 1
				else
					idle_time = 0
				end
				prev_mario_x = marioX
				prev_mario_y = marioY

				--update joystick#
				lastOutput = readOutputFile()
				setController(lastOutput)


				writeStateToFile(id, controller_state, marioX, marioY, levelID)

				id = id + 1

				--If mario is idle for 5 seconds, restart
				if idle_time > 300 then
					setup()
				end
			else
				setup()
			end
		end
		emu.frameadvance()
	end
end

do_run = false
function startExperiment()
	experiment_id = tonumber(forms.gettext(form_exp_id))
	if not do_run then
		setup()
		os.execute("start pythonw.exe watch.py -e " .. experiment_id)
		io.popen("start python.exe train.py --model_path experiments/experiment_" .. experiment_id .. "/models/ --data_path experiments/experiment_" .. experiment_id .. "/data --num_workers 2 --sleep 10")
		forms.settext(form_start_button, "Stop")
		do_run = true
	else
		os.execute("taskkill /IM pythonw.exe /F")
		forms.settext(form_start_button, "Start")
		do_run = false
	end
end

function makeForm()
	form = forms.newform(500, 500, "nn_mario")
	form_welcome_message = forms.label(form, "Random label", 5, 8)
	form_error_messages = forms.label(form, "", 5, 33)
	
	form_id_text = forms.label(form, "Experiment ID", 5, 58)
	form_exp_id = forms.textbox(form, "0", 100, 20, "UNSIGNED", 120, 58)
	
	form_start_button = forms.button(form, "Start", startExperiment, 120, 83, 100, 20)
end

makeForm()
mainLoop()