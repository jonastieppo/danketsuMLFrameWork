def argumentTypeChecker(expectedType, receivedType):
    '''
    check if the argument is valid
    '''
    if type(receivedType) != expectedType:
        raise Exception(f"Argument invalid. Received {type(receivedType)}, but expected {expectedType}")
    else:
        return True
    
def argumentChecker(poolOfArguments : dict, expectedArgument):
    '''
    Checks if the method received the correct arguments. It will print the expeted ones
    '''

    def print_args():
        stream = "\n--- Arguments Received----\n"
        for each_arg in poolOfArguments.keys():
            stream+=f"* {each_arg}\n"
        return stream

    if expectedArgument not in poolOfArguments.keys():
        raise Exception(fr"Expeted argument {expectedArgument}, but received: {print_args()}")