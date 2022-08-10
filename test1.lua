time_stamp = os.time(os.date("!*t"))
id = 0

FRAMES_PER_SCREENSHOT = 10
STATE_FILE_NAME = "data/data_" .. time_stamp .. "/stateFile.csv"
BUTTONS_FILE_NAME = "data/data_" .. time_stamp .. "/outputFile.csv"

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

	--local layer1x = mainmemory.read_s16_le(BYTE_LAYER1X);
	--local layer1y = mainmemory.read_s16_le(BYTE_LAYER1Y);

	--local screenX = marioX-layer1x
	--local screenY = marioY-layer1y

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

function setup()
	--Make state file
	os.execute("mkdir data")
	os.execute("cd data && mkdir data_" .. time_stamp)

	local stateFile = io.open(STATE_FILE_NAME, "w")
	stateFile:write("id,A,B,X,Y,Up,Down,Left,Right,MarioX,MarioY,LevelID\n")
	io.close(stateFile)

	--Setup other stuff
	frame_counter = mainmemory.read_u8(BYTE_EFFECTIVE_FRAME_COUNTER)
	load_level = false
end

function checkInLevel()
	--Update frame counter
	--If game mode is 0 (in level) and frame counter is running, return true
	prev_frame_counter = frame_counter
	frame_counter = mainmemory.read_u8(BYTE_EFFECTIVE_FRAME_COUNTER)
	game_mode = mainmemory.read_u8(BYTE_GAME_MODE)

	--if level_id is not 0, update it (this value only flashes at start of level)
	check_levelID = mainmemory.read_u8(BYTE_LEVEL_ID)

	bool_in_level = frame_counter == prev_frame_counter + 1 and game_mode == 0

	--print(game_mode)

	--update level id only at loading of the level
	if check_levelID > 0 and load_level == false then
		print(game_mode)
		load_level = true
		levelID = check_levelID
		print('Load level ' .. levelID)
	elseif check_levelID == 0 then
		load_level = false
	end

	return bool_in_level
end

function split(inputstr, sep)
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
	t = split(inputstr, "\n")
	return t[#t]
end

function readOutputFile()
	local f = io.open(BUTTONS_FILE_NAME, "r")
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
	if not isempty(line) then
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
		if checkInLevel() then
			if id % FRAMES_PER_SCREENSHOT == 0 then
				client.screenshot("data/data_" .. time_stamp .. "/screenshot_" .. id .. ".png")
			end
		
			controller_state = getController()
			marioX, marioY = getPositions()

			--update joystick#
			lastOutput = readOutputFile()
			setController(lastOutput)


			writeStateToFile(id, controller_state, marioX, marioY, levelID)

			id = id + 1
		end
		emu.frameadvance()
	end
end

setup()
mainLoop()