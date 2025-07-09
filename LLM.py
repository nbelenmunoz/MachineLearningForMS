from openai import OpenAI
import os
import base64
from tabulate import tabulate
import pandas as pd
import json
import re
import math
from datetime import datetime


#Configuration
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY', 'sLyu_yHqkA'))

MODEL = "gpt-4.1-mini"
print(f"Using model: {MODEL}")

gpt_cache = {}

GCODE_CONFIG = {
    'unit': 'G21',        # Millimeter mode
    'absolute': 'G90',    # Absolute positioning
    'feed_mode': 'G94',   # Units per minute
    'safe_z': 20.0,       # Safe Z height (mm)
    'plunge_feed': 30,    # Plunge feed rate (% of normal feed)
    'rapid_feed': 5000,   # Rapid movement feed (mm/min)
    'precision': 3,       # Decimal places
}

MAX_SPEED_RPM = 15_000  
# Material database with expanded parameters
MATERIAL_DB = {
    'aluminium': {
        'sfm': 300,
        'chipload': {2: 0.013, 3: 0.020, 4: 0.027, 6: 0.032, 8: 0.043, 10: 0.050},
        'color': 'yellow'
    },
    'steel': {
        'sfm': 80,
        'chipload': {2: 0.005, 3: 0.008, 4: 0.010, 6: 0.015, 8: 0.020, 10: 0.025},
        'color': 'red'
    },
    'stainless': {
        'sfm': 60,
        'chipload': {2: 0.004, 3: 0.006, 4: 0.008, 6: 0.012, 8: 0.016, 10: 0.020},
        'color': 'blue'
    },
    'titanium': {
        'sfm': 50,
        'chipload': {2: 0.003, 3: 0.005, 4: 0.007, 6: 0.010, 8: 0.013, 10: 0.016},
        'color': 'purple'
    },
    'brass': {
        'sfm': 200,
        'chipload': {2: 0.015, 3: 0.023, 4: 0.030, 6: 0.036, 8: 0.045, 10: 0.052},
        'color': 'gold'
    },
    'plastic': {
        'sfm': 150,
        'chipload': {2: 0.010, 3: 0.015, 4: 0.020, 6: 0.025, 8: 0.030, 10: 0.035},
        'color': 'green'
    },
    'wood': {
        'sfm': 1000,
        'chipload': {2: 0.017, 3: 0.025, 4: 0.033, 6: 0.040, 8: 0.053, 10: 0.060},
        'color': 'brown'
    }
}

TOOL_DB = {
    'drill bit': {
        'flutes': 2,
        'type': 'drilling',
        'gcode_tool_change': 'M6 T{tool_number}',
        'approach_template': 'G0 X{x} Y{y}\nG81 Z{z} R{z_approach} F{feed}',
        'color': 'blue',
        'max_doc': 0.4  # Depth of cut multiplier
    },
    'end mill': {
        'flutes': 4,
        'type': 'milling',
        'gcode_tool_change': 'M6 T{tool_number}',
        'approach_template': 'G0 X{x} Y{y}\nG0 Z{z_approach}\nG1 Z{z} F{plunge_feed}',
        'color': 'red',
        'max_doc': 0.5
    },
    'face mill': {
        'flutes': 6,
        'type': 'facing',
        'gcode_tool_change': 'M6 T{tool_number}',
        'approach_template': 'G0 X{x} Y{y}\nG0 Z{z_approach}\nG1 Z{z} F{plunge_feed}',
        'color': 'green',
        'max_doc': 0.6
    },
    'ball nose': {
        'flutes': 4,
        'type': 'contouring',
        'gcode_tool_change': 'M6 T{tool_number}',
        'approach_template': 'G0 X{x} Y{y}\nG0 Z{z_approach}\nG1 Z{z} F{plunge_feed}',
        'color': 'purple',
        'max_doc': 0.3
    },
    'slot drill': {
        'flutes': 2,
        'type': 'slotting',
        'gcode_tool_change': 'M6 T{tool_number}',
        'approach_template': 'G0 X{x} Y{y}\nG0 Z{z_approach}\nG1 Z{z} F{plunge_feed}',
        'color': 'orange',
        'max_doc': 0.4
    },
    'chamfer tool': {
        'flutes': 1,
        'type': 'chamfering',
        'gcode_tool_change': 'M6 T{tool_number}',
        'approach_template': 'G0 X{x} Y{y}\nG0 Z{z_approach}\nG1 Z{z} F{plunge_feed}',
        'color': 'cyan',
        'max_doc': 0.2
    },
    'tap': {
        'flutes': 4,
        'type': 'threading',
        'gcode_tool_change': 'M6 T{tool_number}',
        'approach_template': 'G0 X{x} Y{y}\nG84 Z{z} R{z_approach} F{pitch}',
        'color': 'yellow',
        'max_doc': 1.0
    },
    'reamer': {
        'flutes': 6,
        'type': 'finishing',
        'gcode_tool_change': 'M6 T{tool_number}',
        'approach_template': 'G0 X{x} Y{y}\nG85 Z{z} R{z_approach} F{feed}',
        'color': 'pink',
        'max_doc': 0.1
    },
    'boring bar': {
        'flutes': 1,
        'type': 'boring',
        'gcode_tool_change': 'M6 T{tool_number}',
        'approach_template': 'G0 X{x} Y{y}\nG76 Z{z} R{z_approach} Q{doc} F{feed}',
        'color': 'brown',
        'max_doc': 0.05
    }
}


#GPT CALLS
def gpt_call(system_prompt, user_query, temperature=0.1, max_tokens=2000):
    key = (MODEL, system_prompt, user_query, temperature, max_tokens)
    if key in gpt_cache:
        print("Using cached GPT response")
        return gpt_cache[key]
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        content = response.choices[0].message.content
        gpt_cache[key] = content
        return content
    except Exception as e:
        print(f"GPT API Error: {str(e)}")
        sys.exit(1)

#IMAGE PROCESSING

def gpt_image(query='', image_path='', temperature=0.1, max_tokens=3000):
  with open(image_path, "rb") as image_file:
      encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

    system_prompt = (
        "Analyze the technical drawing provided and return a complete description of the component. "
        "You have one or more technical drawings and section drawings of a mechanical part "
        "The drawing includes clear dimensional markings. "
        "Provide a concise but comprehensive geometric description of the part in precise technical language. "
        "Explicitly reference all given dimensions in the drawing. Include overall shape and principal dimensions "
        "(length, width, thickness); clearly identify and describe the location, orientation, and size of all visible features "
        "such as holes, slots, pockets, chamfers, and fillets. "
        "State exactly on which face each feature is located (e.g., 'top face', 'side face'), using provided dimensions."
        "If a feature is centered on a face, note that explicitly. "
        "Do not overlook any geometric dimensions in the technical drawings. Do not invent or assume features not clearly shown or dimensioned. "
        "Avoid manufacturing advice, tooling suggestions, or markup annotations. Output only the geometric description."
    )


      multimodal_content = [
          {"type": "text", "text": query},
          {
              "type": "image_url",
              "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}
          },
      ]

      messages = [
          {"role": "system", "content": system_prompt},
          {"role": "user",   "content": multimodal_content}
      ]

      response = client.chat.completions.create(
          model=MODEL,
          messages=messages,
          temperature=temperature,
          max_tokens=max_tokens
      )
      return response.choices[0].message.content

#Text and plan process

def call_process_plan(part_name, material_desc, machine, part_desc, dimensions, tolerance, surface):
    system_prompt = (
        "You are an Expert CNC Process Engineer. "
        "Your role is to:\n"
        " - Analyze detailed part drawings or descriptions, including geometry, tolerances, surface finish, and material.\n"
        " - Propose an optimized sequence of machining operations (roughing, finishing, drilling, tapping, etc.) tailored to the part’s features.\n"
        " - Specify machine setups: workholding methods, fixturing orientation, datum references, and work offsets.\n"
        " - Recommend cutting tools (end mills, drills, inserts, holders), tool materials/coatings, and toolpaths (2.5D, 3-axis, 4-axis, 5-axis).\n"
        " - Define cutting parameters: spindle speeds, feed rates, depths of cut, and coolant strategy.\n"
        " - Estimate cycle time, material removal rates, and tooling cost considerations.\n"
        " - Incorporate quality control checks: critical dimensions, inspection features, and measurement methods.\n"
        " - Advise on CAM software workflow, G-code generation, and post-processor settings.\n"
        " - Highlight any safety, maintenance, or compliance issues (machine limits, tool deflection, vibrations, material hazards).\n"
        "Present your output in a clear, step-by-step format, and include any assumptions you make."
    )

    user_query = (
        f"Part name: {part_name}. "
        f"Raw material: {material_desc}. "
        f"Machine: {machine}. "
        f"Features: {part_desc}. Dimensions: {dimensions}. Tolerance: {tolerance}. Surface finish: {surface}. "
        "Please provide a brief interpretation of the part and its critical features; "
        "A setup plan detailing how to fixture the raw material; "
        "A process plan with steps (roughing, drilling, finishing); "
        "For each step include: setup, tool (size/type), spindle speed (RPM), feed rate (mm/min), remarks; "
        "Return as JSON array of objects with keys ['step','operation','setup','tool','speed','feed','remarks']. "
        "Also provide a draft G-code snippet for the CNC machine."
    )

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ],
        temperature=0.2,
        max_tokens=2000
    )
    return response.choices[0].message['content']

#Json
def parse_output(raw):
    try:
        # First try extracting JSON
        json_match = re.search(r'(\[{.*?}\])', raw, re.DOTALL)
        if json_match:
            return pd.DataFrame(json.loads(json_match.group(1)))
        
        # Fallback to direct parse
        return pd.DataFrame(json.loads(raw))
    except:
        print("JSON Parsing Failed. Raw response:")
        print(raw)
        return None

def calculate_parameters(material, tool_type, diameter, operation):
    """Calculate recommended machining parameters"""
    material_data = MATERIAL_DB.get(material.lower(), {})
    tool_data = TOOL_DB.get(tool_type.lower(), {})
    
    if not material_data or not tool_data:
        return None, None, None

    # Calculate recommended surface speed (RPM)
    sfm = material_data.get('sfm', 100)
    rpm = max(500, min(MAX_SPEED_RPM, int((sfm * 304.8) / (math.pi * diameter / 10))))
    
    # Calculate recommended feed rate
    flutes = tool_data.get('flutes', 4)
    chipload = material_data['chipload'].get(
        diameter, 
        material_data['chipload'][max(material_data['chipload'].keys())]
    )
    feed = rpm * flutes * chipload
    
    # Calculate depth of cut based on operation type
    if 'rough' in operation.lower():
        depth = min(0.4 * diameter, 5.0)  
    elif 'finish' in operation.lower():
        depth = min(0.1 * diameter, 1.0)  
    else:
        depth = min(0.2 * diameter, 3.0)  
    
    return rpm, feed, depth

def validate_step(step, material):
    errors = []
    
    # Extract tool information
    tool_match = re.match(r"(.+?),\s*ø\s*([\d.]+)", step['tool'])
    if not tool_match:
        errors.append("Invalid tool format")
        return errors
    
    tool_type = tool_match.group(1).strip()
    diameter = float(tool_match.group(2))
    
    # Check for known tool type
    if tool_type.lower() not in TOOL_DB:
        errors.append(f"Unrecognized tool: {tool_type}")
    
    # Validate numerical parameters
    try:
        speed = int(step['speed'])
        feed = float(step['feed'])
        depth = float(step['depth'])
        
        if speed <= 0 or speed > MAX_SPEED_RPM:
            errors.append(f"Invalid RPM: {speed} (Max {MAX_SPEED_RPM})")
        if feed <= 0:
            errors.append(f"Invalid feed: {feed}")
        if depth <= 0:
            errors.append(f"Invalid depth: {depth}")
        
        # Depth of cut validation
        if depth > 0.5 * diameter:
            errors.append(f"Depth {depth}mm > 50% tool diameter")
            
        # Parameter recommendation check
        rec_rpm, rec_feed, rec_depth = calculate_parameters(
            material, tool_type, diameter, step['operation']
        )
        
        if rec_rpm:
            rpm_diff = abs(speed - rec_rpm) / rec_rpm
            if rpm_diff > 0.3:
                errors.append(f"RPM differs >30% from recommendation ({rec_rpm})")
                
            feed_diff = abs(feed - rec_feed) / rec_feed
            if feed_diff > 0.4:
                errors.append(f"Feed differs >40% from recommendation ({rec_feed:.1f})")
            
            if depth > rec_depth * 1.5:
                errors.append(f"Depth exceeds recommendation ({rec_depth:.2f}mm)")
                
    except (ValueError, TypeError):
        errors.append("Invalid numerical parameters")
    
    return errors


def display(df, issues):
    print("\n=== CNC Machining Process Plan ===")
    for _, row in df.iterrows():
        print(f"\nStep {row['step']}: {row['operation']}")
        print("-"*40)
        print(f"Setup:     {row['setup']}")
        print(f"Tool:      {row['tool']}")
        print(f"Speed:     {row['speed']} RPM")
        print(f"Feed Rate: {row['feed']} mm/min")
        if row.get('remarks'):
            print(f"Remarks:   {row['remarks']}")

    if issues:
        print("\n=== Validation Warnings ===")
        # Tabulate structured issues
        print(tabulate(issues, headers=['step','field','message'], tablefmt='grid'))
    else:
        print("\nNo validation issues detected.")


def summary_table(df):
    print("\n=== Process Plan Summary Table ===")
    table = df[['step','operation','setup','tool','speed','feed','remarks']].copy()
    table['speed'] = table['speed'].astype(str) + " RPM"
    table['feed']  = table['feed'].astype(str)  + " mm/min"
    print(tabulate(table, headers='keys', tablefmt='grid', showindex=False))

def generate_gcode(plan, origin=(0, 0, 0)):
    """
    Generate complete G-code program from machining plan
    Returns: (gcode_string, warnings)
    """
    gcode = []
    warnings = []
    tool_map = {}
    current_tool = None
    
    # Program header
    gcode.append(f"; CNC Program for {plan['part_name']}")
    gcode.append(f"; Material: {plan['material']}")
    gcode.append(f"; Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    gcode.append(f"{GCODE_CONFIG['unit']} ; Millimeter mode")
    gcode.append(f"{GCODE_CONFIG['absolute']} ; Absolute positioning")
    gcode.append(f"{GCODE_CONFIG['feed_mode']} ; Feed per minute")
    gcode.append("G17 ; XY plane selection")
    gcode.append("G49 ; Cancel tool length compensation")
    gcode.append("G80 ; Cancel canned cycles")
    gcode.append("\n")
    
    # Set coordinate system
    gcode.append(f"G54 ; Work coordinate system")
    gcode.append(f"G0 Z{GCODE_CONFIG['safe_z']:.{GCODE_CONFIG['precision']}f} ; Move to safe Z")
    gcode.append("\n")
    
    # Process each operation
    for i, op in enumerate(plan['operations']):
        # Parse tool information
        tool_match = re.match(r"(.+?),\s*ø\s*([\d.]+)", op['tool'])
        if not tool_match:
            warnings.append(f"Step {op['step']}: Invalid tool format '{op['tool']}'")
            continue
            
        tool_name = tool_match.group(1).strip().lower()
        diameter = float(tool_match.group(2))
        
        # Get tool data
        tool_data = TOOL_DB.get(tool_name)
        if not tool_data:
            warnings.append(f"Step {op['step']}: Unsupported tool '{tool_name}'")
            continue
        
        # Assign tool number (first occurrence)
        if tool_name not in tool_map:
            tool_number = len(tool_map) + 1
            tool_map[tool_name] = tool_number
        else:
            tool_number = tool_map[tool_name]
        
        # Tool change if needed
        if current_tool != tool_number:
            gcode.append(f"; Tool change: {tool_name} Ø{diameter}mm")
            gcode.append(tool_data['gcode'].format(tool_number=tool_number))
            gcode.append(f"S{int(op['speed'])} M3 ; Spindle on CW")
            gcode.append(f"G43 H{tool_number} ; Tool length compensation")
            gcode.append(f"G0 Z{GCODE_CONFIG['safe_z']:.{GCODE_CONFIG['precision']}f}")
            current_tool = tool_number
            gcode.append("\n")
        
        # Operation header
        gcode.append(f"; Step {op['step']}: {op['operation']}")
        
        # Generate toolpath based on operation type
        op_type = tool_data['type']
        if op_type == 'milling':
            path, op_warnings = generate_milling_path(op, diameter)
            gcode.append(path)
            warnings.extend(op_warnings)
        elif op_type == 'drilling':
            path, op_warnings = generate_drilling_path(op, diameter)
            gcode.append(path)
            warnings.extend(op_warnings)
        else:
            warnings.append(f"Step {op['step']}: No path generation for {op_type} operations")
            gcode.append(f"; Manual {op_type} operation required")
        
        gcode.append("\n")
    
    # Program footer
    gcode.append("; Program completion")
    gcode.append("M5 ; Spindle stop")
    gcode.append(f"G0 Z{GCODE_CONFIG['safe_z']:.{GCODE_CONFIG['precision']}f} ; Retract")
    gcode.append("G28 G91 Z0 ; Return to Z home")
    gcode.append("G28 G91 X0 Y0 ; Return to XY home")
    gcode.append("M30 ; Program end")
    
    return "\n".join(gcode), warnings

def generate_milling_path(operation, tool_diameter):
    """Generate milling toolpath for common operations"""
    path = []
    warnings = []
    
    # Extract parameters
    depth = float(re.search(r"[\d.]+", str(operation['depth'])).group())
    feed = float(re.search(r"[\d.]+", str(operation['feed'])).group())
    plunge_feed = feed * (GCODE_CONFIG['plunge_feed'] / 100)
    
    # Approach parameters
    approach_params = {
        'x': 0,
        'y': 0,
        'z': -depth,
        'z_approach': 2.0,  # 2mm above material
        'feed': feed,
        'plunge_feed': plunge_feed
    }
    
    # Generate basic rectangle pocket (simplified)
    if 'pocket' in operation['operation'].lower():
        width = 20  # Example dimensions - should come from CAD
        height = 30
        stepover = tool_diameter * 0.6
        
        path.append("; Pocket milling operation")
        path.append(TOOL_DB['end mill']['approach'].format(**approach_params))
        
        # Basic pocketing pattern
        path.append(f"G1 X{width/2:.3f} Y{height/2:.3f} F{feed}")
        path.append(f"G1 Z-{depth:.3f} F{plunge_feed}")
        # ... (actual pocketing toolpath would go here)
        path.append("; Pocket toolpath would be generated based on CAD geometry")
        
    elif 'contour' in operation['operation'].lower():
        path.append("; Contour milling operation")
        path.append(TOOL_DB['end mill']['approach'].format(**approach_params))
        # ... (contour toolpath generation)
        path.append("; Contour toolpath would follow part geometry")
    
    else:
        warnings.append(f"Generic milling operation: {operation['operation']}")
        path.append("; Standard milling approach")
        path.append(TOOL_DB['end mill']['approach'].format(**approach_params))
        path.append("; Specific toolpath requires CAM software")
    
    return "\n".join(path), warnings

def generate_drilling_path(operation, tool_diameter):
    path = []
    warnings = []
    
    depth = float(re.search(r"[\d.]+", str(operation['depth'])).group())
    feed = float(re.search(r"[\d.]+", str(operation['feed'])).group())
    
    hole_positions = [
        (10, 10),
        (10, 20),
        (20, 10),
        (20, 20)
    ]
    
    path.append("; Drilling operation")
    path.append(f"G98 ; Return to initial Z")
    
    for i, (x, y) in enumerate(hole_positions):
        path.append(f"; Hole {i+1} at X{x} Y{y}")
        path.append(f"G0 X{x} Y{y}")
        
        # Drilling cycle (G81)
        cycle_params = {
            'x': x,
            'y': y,
            'z': -depth - 1,  # Drill through
            'z_approach': 5.0,
            'feed': feed
        }
        path.append(TOOL_DB['drill bit']['approach'].format(**cycle_params))
    
    path.append("G80 ; Cancel drilling cycle")
    return "\n".join(path), warnings

def save_gcode(gcode, filename):
    """Save G-code to file with validation"""
    if not filename.endswith('.nc'):
        filename += '.nc'
    
    with open(filename, 'w') as f:
        f.write(gcode)
    print(f"G-code saved to {filename}")
    
    # Basic validation
    line_count = len(gcode.split('\n'))
    tool_changes = gcode.count('M6')
    warnings = []
    
    if line_count < 20:
        warnings.append(f"Short program ({line_count} lines) - verify completeness")
    if 'M30' not in gcode:
        warnings.append("Missing program end command (M30)")
    if 'G21' not in gcode:
        warnings.append("Missing metric units specification (G21)")
    
    return warnings

def reflect():
    print("""
    ### Reflection
    - Verify speeds/feeds against tooling specs.
    - Check critical dimensions in CAD/CAM.
    - Adapt JSON for downstream CAM import as needed.
    - Always simulate before real cuts.
    """ )

def main():
    print("CNC Process Planning Assistant")
    print("="*60)
    part_name = input("\nPart name: ").strip()
    material = material_selector()
    
    print("\nMachine Type:")
    print("1. CNC Milling Machine")
    print("2. CNC Lathe")
    print("3. 5-Axis Machining Center")
    machine_choice = input("Select machine type (1-3): ").strip()
    machine = ["CNC Milling Machine", "CNC Lathe", "5-Axis Machining Center"][int(machine_choice)-1] if machine_choice in "123" else "CNC Milling Machine"
    
    print("\nInput Method:")
    print("1. Upload technical drawing (image)")
    print("2. Enter text description")
    input_method = input("Select input method (1-2): ").strip()
    
    if input_method == "1":
        image_path = input("Image file path: ").strip()
        part_desc = gpt_image(image_path)
        if not part_desc:
            print("Using fallback text description")
            part_desc = input("Describe part features: ").strip()
    else:
        part_desc = input("Describe part features (dimensions, holes, slots, tolerances):\n").strip()
    
    print("\n Generating process plan...")
    json_plan = call_process_plan(part_name, material, machine, part_desc)
    
    try:
        plan_data = json.loads(json_plan)
    except json.JSONDecodeError:
        print("Failed to parse process plan JSON")
        print("Raw response:\n", json_plan)
        return
    

    validation_issues = {}
    for i, step in enumerate(plan_data['operations']):
        errors = validate_step(step, material)
        if errors:
            validation_issues[i+1] = errors

    final_plan = {
        'part_name': part_name,
        'material': material,
        'machine': machine,
        'interpretation': plan_data.get('interpretation', ''),
        'workholding': plan_data.get('workholding', ''),
        'operations': plan_data['operations'],
        'validation_issues': validation_issues
    }
    
    display_plan(final_plan)
    
    if final_plan['operations']:
        gen_gcode = input("\nGenerate G-code program? (y/n): ").lower()
        if gen_gcode == 'y':
            print("\n Generating G-code...")
            gcode, warnings = generate_gcode(final_plan)
            
            print("\n=== G-CODE PREVIEW ===")
            print("\n".join(gcode.split('\n')[:15]) + "\n...\n")
            
            if warnings:
                print(" G-CODE GENERATION WARNINGS:")
                for warn in warnings:
                    print(f"  - {warn}")
            
            save = input("\n Save G-code to file? (y/n): ").lower()
            if save == 'y':
                filename = f"{final_plan['part_name'].replace(' ', '_')}_program.nc"
                save_warnings = save_gcode(gcode, filename)
                
                if save_warnings:
                    print("\n G-CODE VALIDATION NOTES:")
                    for warn in save_warnings:
                        print(f"  - {warn}")
    
if __name__ == "__main__":
    main()
