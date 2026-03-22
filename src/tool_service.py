from robot_functions import Robot


class ToolService():
    def __init__(self, robot: Robot):
        self.available_tools = {
            "sort_all_bricks": robot.sort_bricks,
            "get_collision_free_bricks": robot.display_collision_free_bricks,
            "grab_brick": robot.grab_brick_by_center_point,
            "put_down_brick": robot.put_down_brick,
            "get_all_bricks": robot.display_bricks,
            "place_brick": robot.place_bricks,
            "move_arm": robot.simple_move,
            "turn_arm": robot.full_turn
        }

    def parse_and_execute_response(self, tools):
        function_name = tools['function_name'][0]
        function_arg = tools['arguments'][0]
        if not function_name:
            print(tools, " is not correct.")
            return
        function_to_call = self.available_tools.get(function_name, None)
        if function_to_call is None:
            print(function_to_call, " is not a defined function")
            return
        print(f"calling function {function_name} with arguments {function_arg}")
        one_arg = {"grab_brick", "sort_all_bricks", "move_arm", "turn_arm"}
        if function_name in one_arg:
            success = function_to_call(function_arg)
        elif function_name == "place_brick":
            function_arg_2 = tools['arguments'][1]
            success = function_to_call(function_arg, function_arg_2)
        else:
            success = function_to_call()
        return success

           
